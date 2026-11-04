"""Independent V2 self-play-only rare visible-wood outcome correction.

NEVER rewrites original Q/V9 files. Uses *only* source-checkpoint-committed
player-observation `.npz` from unmodified Minecraft worlds, and a previous
self-trained Q checkpoint as an action-distribution anchor. No scripts or
privileged block geometry are involved. New model is evaluated separately.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.experience_archive import archive_path,read_episode
from gmaincraft.natural_online_policy import TEMPORAL_V9_SCHEMA,MOTOR_V7_HEAD_SIZES
from gmaincraft.fresh_seed_schedule import SCHEMA as STREAM_SCHEMA
from gmaincraft.replay_q_skill import SCHEMA as LEGACY_SCHEMA,CRACK_GOAL
from gmaincraft.replay_q_rare_v2 import (RareOutcomeQSkill,SCHEMA,WOOD_CRACK_GOAL,
                                          WOOD_INDEX,wood_crack_transitions)


def cli(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-run',required=True,help='Completed original V9 collector with true own observations')
    ap.add_argument('--anchor-model',required=True,help='Older self-trained Q snapshot; never external demonstrations')
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--epochs',type=int,default=25,help='CUMULATIVE independent correction epochs')
    ap.add_argument('--updates-per-epoch',type=int,default=25)
    ap.add_argument('--batch-size',type=int,default=48)
    ap.add_argument('--max-episodes',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=20261104)
    ap.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    ap.add_argument('--anchor-weight',type=float,default=.2,
                    help='nonwood player-state KL to previous self-trained baseline')
    ap.add_argument('--success-weight',type=float,default=.6,
                    help='own successful observed-action imitation weight')
    args=ap.parse_args(argv)
    for tag in (args.source_run,args.anchor_model,args.run_tag):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,22}',tag):ap.error('Unsafe run tag')
    if (not 1<=args.epochs<=100000 or not 1<=args.updates_per_epoch<=10000 or
        not 12<=args.batch_size<=256 or args.batch_size%3 or
        not 1<=args.max_episodes<=10000 or not 0<=args.seed<2**63 or
        not 0<=args.anchor_weight<=10 or not 0<args.success_weight<=10):
        ap.error('Invalid replay sampling/training budget')
    return args


def read_committed_source(root,run,max_episodes):
    import torch
    source=torch.load(root/'checkpoints'/f'fresh_stream_{run}.pt',map_location='cpu',weights_only=True)
    meta=source['metadata']
    if (source.get('stream')!=STREAM_SCHEMA or
        meta.get('policy_schema')!=TEMPORAL_V9_SCHEMA or
        meta.get('reward_profile')!='wood_temporal_v9' or
        tuple(meta.get('policy_action_heads',()))!=MOTOR_V7_HEAD_SIZES or
        meta.get('kind') not in ('gmaincraft_q_collect_v1','gmaincraft_q_collect_v2_rare_outcome','gmaincraft_fresh_stream_v2_fixed_spawn') or
        type(source.get('next_episode')) is not int or source['next_episode']<1):
        raise ValueError('Refusing noncommitted or incompatible player-only V9 source')
    cursor=source['next_episode']
    ep_indices=[]
    rows=[]
    for i in range(max(0,cursor-max_episodes),cursor):
        path=archive_path(root/'data-local'/'experience',run,i)
        if not path.is_file() or path.is_symlink():continue
        emeta,data=read_episode(path)
        if emeta['observation_schema']!=TEMPORAL_V9_SCHEMA:
            raise ValueError('Mixed replay observation schemas are not valid')
        rows.extend(wood_crack_transitions(emeta,data))
        ep_indices.append(i)
    positive=[r for r in rows if r['achieved']]
    wood_no_progress=[r for r in rows if not r['achieved'] and r['before_own'][WOOD_INDEX]>.5]
    other=[r for r in rows if not r['achieved'] and r['before_own'][WOOD_INDEX]<=.5]
    if not positive or not wood_no_progress or not other:
        raise ValueError('Need genuine target progress AND varied observed failures/other states')
    stats=dict(source_run=run,source_committed_cursor=cursor,source_seed_root=meta['seed_root'],
               episodes_scanned=len(ep_indices),original_actions=len(rows),
               genuine_observed_wood_crack_outcomes=len(positive),
               actual_wood_crosshair_nonprogress_actions=len(wood_no_progress),
               other_actions=len(other),
               original_episode_indices_first_last=[ep_indices[0],ep_indices[-1]])
    return (positive,wood_no_progress,other),stats


def save_atomic(path,model,metadata,epoch):
    staged=path.with_suffix('.pt.pending')
    if staged.exists():raise FileExistsError('Interrupted V2 checkpoint staging exists; inspect it')
    # Atomic staging and rename after CPU+GPU model/optimizer/RNG are saved.
    with staged.open('xb') as stream:
        model.torch.save(dict(schema=SCHEMA,metadata=metadata,model=model.state(),epochs=epoch),stream)
        stream.flush();os.fsync(stream.fileno())
    os.replace(staged,path)


def train_epoch(model,bins,*,seed,updates,batch_size,anchor_weight=.2,success_weight=.6):
    rng=np.random.default_rng(seed)
    reports=[]
    for _ in range(updates):
        # Maintain a permanent stratified rare-success reservoir across ALL
        # source worlds, unlike the previous most-recent-32-world window.
        # This is TRAINING sampling only; the live policy sees no examples,
        # success labels, hidden target coordinates or source episode index.
        batch=[]
        for pool in bins:
            selected=rng.integers(0,len(pool),size=batch_size//3)
            batch.extend(pool[int(i)] for i in selected)
        rng.shuffle(batch)
        reports.append(model.update(batch,WOOD_CRACK_GOAL,anchor_weight=anchor_weight,
                                    success_weight=success_weight))
    return dict(loss=round(float(np.mean([r['loss'] for r in reports])),6),
                td=round(float(np.mean([r['td'] for r in reports])),6),
                imitation=round(float(np.mean([r['positive_imitation'] for r in reports])),6),
                anchor_kl=round(float(np.mean([r['anchor_kl'] for r in reports])),6),
                updates=model.updates)


def main(argv=None):
    args=cli(argv)
    import torch
    ckpt=ROOT/'checkpoints'/f'replay_q_{args.run_tag}.pt'
    journal=ROOT/'logs'/f'replay_q_{args.run_tag}.jsonl'
    lock_path=ROOT/'checkpoints'/f'replay_q_{args.run_tag}.lock'
    ckpt.parent.mkdir(exist_ok=True);journal.parent.mkdir(exist_ok=True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.resume:
            if not ckpt.exists() or not journal.exists():raise FileNotFoundError('Missing v2 prior model and journal')
        elif ckpt.exists() or journal.exists() or ckpt.with_suffix('.pt.pending').exists():
            raise FileExistsError('V2 run would overwrite an existing or incomplete model')
        bins,stats=read_committed_source(ROOT,args.source_run,args.max_episodes)
        anchor=torch.load(ROOT/'checkpoints'/f'replay_q_{args.anchor_model}.pt',
                          map_location='cpu',weights_only=True)
        if (anchor.get('schema')!=LEGACY_SCHEMA or
            anchor.get('metadata',{}).get('goal')!=CRACK_GOAL or
            anchor['metadata'].get('source_seed_root')!=stats['source_seed_root']):
            raise ValueError('Only same-world-stream self-trained original Q may anchor v2')
        metadata=dict(source_run=args.source_run,source_observation_schema=TEMPORAL_V9_SCHEMA,
                      action_sizes=MOTOR_V7_HEAD_SIZES,goal=WOOD_CRACK_GOAL,
                      source_seed_root=stats['source_seed_root'],method=SCHEMA,
                      anchor_model=args.anchor_model,seed=args.seed,
                      updates_per_epoch=args.updates_per_epoch,batch_size=args.batch_size,
                      max_episodes=args.max_episodes,anchor_weight=args.anchor_weight,
                      success_weight=args.success_weight,
                      anchor_is_own_previously_trained_network=True)
        model=RareOutcomeQSkill(seed=args.seed,device=args.device)
        epoch=0
        if args.resume:
            previous=torch.load(ckpt,map_location=model.device,weights_only=True)
            if previous['schema']!=SCHEMA or previous['metadata']!=metadata:
                raise ValueError('Unsafe v2 checkpoint/resume schema drift')
            model.restore(previous['model'])
            epoch=previous['epochs']
        else:model.initialize_from_selfplay(anchor)
        with journal.open('a' if args.resume else 'x') as stream:
            stream.write(json.dumps(dict(stage='data_snapshot',current_epoch=epoch,**stats),sort_keys=True)+'\n');stream.flush()
            print('RARE Q DATA',json.dumps(stats,sort_keys=True),flush=True)
            for e in range(epoch,args.epochs):
                start=time.monotonic()
                result=train_epoch(model,bins,seed=args.seed+e,updates=args.updates_per_epoch,
                                   batch_size=args.batch_size,anchor_weight=args.anchor_weight,
                                   success_weight=args.success_weight)
                save_atomic(ckpt,model,metadata,e+1)
                row=dict(stage='epoch_committed',epoch=e+1,
                         source_cursor=stats['source_committed_cursor'],
                         wall_s=round(time.monotonic()-start,3),**result)
                stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush()
                print('RARE Q EPOCH',json.dumps(row,sort_keys=True),flush=True)
            print('RARE Q SAVED',ckpt,flush=True)

if __name__=='__main__':main()
