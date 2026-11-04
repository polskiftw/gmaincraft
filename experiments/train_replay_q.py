"""Conservative offline Q-learning from REAL committed V9 Minecraft self-play.

Practice goals: a real observed inventory increase ('minecraft:ITEM') or the
player's own observed crack-stage progression ('effect:visible_crack_progress').
No expert moves, privileged target labels, Minecraft seeds or unseen state are
fed to the model. This is an experimental data-efficiency BASELINE; it cannot
claim mastery without frozen, disjoint natural-game comparisons.
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
from gmaincraft.experience_archive import archive_path,read_episode,ObservedItemGoal
from gmaincraft.replay_q_skill import ReplayQSkill,SCHEMA,CRACK_GOAL,goal_transitions
from gmaincraft.natural_online_policy import TEMPORAL_V9_SCHEMA,MOTOR_V7_HEAD_SIZES
from gmaincraft.fresh_seed_schedule import SCHEMA as STREAM_SCHEMA


def cli(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-run',required=True)
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--goal',required=True,help='Observed minecraft:ITEM or effect:visible_crack_progress')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--epochs',type=int,default=30,help='CUMULATIVE offline dataset passes')
    ap.add_argument('--batch-size',type=int,default=48)
    ap.add_argument('--updates-per-epoch',type=int,default=25)
    ap.add_argument('--max-episodes',type=int,default=2000)
    ap.add_argument('--seed',type=int,default=20261104)
    ap.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    args=ap.parse_args(argv)
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,22}',args.source_run) or
        not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,22}',args.run_tag)):
        ap.error('Unsafe Q training tag')
    if args.goal!=CRACK_GOAL:
        if not args.goal.startswith('minecraft:'):ap.error('Q inventory goals must be vanilla items')
        try:ObservedItemGoal(args.goal)
        except ValueError as exc:ap.error(str(exc))
    if (not 1<=args.epochs<=10000000 or not 1<=args.batch_size<=512 or
        not 1<=args.updates_per_epoch<=10000 or not 1<=args.max_episodes<=10000 or
        not 0<=args.seed<2**63):ap.error('Invalid offline training budget')
    return args


def committed_rows(root,run,goal,max_episodes):
    """Never read in-flight episode files ahead of the actual PPO checkpoint."""
    import torch
    ckpt=root/'checkpoints'/f'fresh_stream_{run}.pt'
    original=torch.load(ckpt,map_location='cpu',weights_only=True)
    meta=original['metadata']
    if (original.get('stream')!=STREAM_SCHEMA or
        meta.get('policy_schema')!=TEMPORAL_V9_SCHEMA or
        meta.get('reward_profile')!='wood_temporal_v9' or
        tuple(meta.get('policy_action_heads',()))!=MOTOR_V7_HEAD_SIZES or
        type(original.get('next_episode')) is not int or original['next_episode']<1):
        raise ValueError('Only committed original V9 motor replay is eligible')
    cursor=original['next_episode']
    start=max(0,cursor-max_episodes)
    directory=root/'data-local'/'experience'/run
    rows=[]
    episodes=0
    outcomes=0
    for i in range(start,cursor):
        path=archive_path(root/'data-local'/'experience',run,i)
        if not path.is_file() or path.is_symlink():continue  # retention may prune old files
        ep_meta,data=read_episode(path)
        if ep_meta['observation_schema']!=TEMPORAL_V9_SCHEMA:raise ValueError('Mixed replay schema')
        records=goal_transitions(ep_meta,data,goal)
        rows.extend(records)
        episodes+=1
        outcomes+=sum(int(r['achieved']) for r in records)
    if not rows:raise ValueError('No committed V9 self-play replay with this run tag')
    if not outcomes:raise ValueError('No genuine achieved outcomes for practice goal; cannot learn from no examples')
    return rows,dict(committed_cursor=cursor,episodes_scanned=episodes,
                     actual_successful_action_outcomes=outcomes,
                     observed_actions=len(rows),goal=goal,source_run=run,
                     source_reward_profile=meta['reward_profile'],
                     source_seed_root=meta['seed_root'])


def save_atomic(path,model,metadata,epoch):
    staged=path.with_suffix('.pt.pending')
    if staged.exists():raise FileExistsError('Stale pending Q checkpoint; inspect before resuming')
    try:
        model.torch.save(dict(schema=SCHEMA,metadata=metadata,
                              model=model.state(),epochs=epoch),staged)
        os.replace(staged,path)
    finally:
        # Preserve incomplete staging file for explicit crash recovery.
        pass


def main(argv=None):
    args=cli(argv)
    ckpt=ROOT/'checkpoints'/f'replay_q_{args.run_tag}.pt'
    journal=ROOT/'logs'/f'replay_q_{args.run_tag}.jsonl'
    ckpt.parent.mkdir(exist_ok=True)
    journal.parent.mkdir(exist_ok=True)
    lock_path=ROOT/'checkpoints'/f'replay_q_{args.run_tag}.lock'
    with lock_path.open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.resume:
            if not ckpt.is_file() or not journal.is_file():
                raise FileNotFoundError('Q model checkpoint + journal required for resume')
        elif ckpt.exists() or journal.exists():
            raise FileExistsError('Q model tag already exists; use --resume or a new tag')
        rows,stats=committed_rows(ROOT,args.source_run,args.goal,args.max_episodes)
        meta=dict(source_run=args.source_run,source_observation_schema=TEMPORAL_V9_SCHEMA,
                  action_sizes=MOTOR_V7_HEAD_SIZES,goal=args.goal,seed=args.seed,
                  method=SCHEMA,source_seed_root=stats['source_seed_root'],
                  updates_per_epoch=args.updates_per_epoch,batch_size=args.batch_size,
                  max_episodes=args.max_episodes)
        model=ReplayQSkill(seed=args.seed,device=args.device)
        epoch=0
        if args.resume:
            old=model.torch.load(ckpt,map_location=model.device,weights_only=True)
            if old['schema']!=SCHEMA or old['metadata']!=meta:raise ValueError('Unsafe replay Q metadata drift')
            model.restore(old['model'])
            epoch=int(old['epochs'])
        weights=np.asarray([r['priority'] for r in rows],np.float64)
        weights/=weights.sum()
        with journal.open('a' if args.resume else 'x',encoding='utf8') as log:
            snapshot=dict(stage='data_snapshot',current_epoch=epoch,**stats)
            log.write(json.dumps(snapshot,sort_keys=True)+'\n');log.flush()
            print('REPLAY Q DATA',json.dumps(snapshot,sort_keys=True),flush=True)
            for e in range(epoch,args.epochs):
                started=time.perf_counter()
                rng=np.random.default_rng(args.seed+e)
                losses=[]
                for _ in range(args.updates_per_epoch):
                    selection=rng.choice(len(rows),size=args.batch_size,replace=True,p=weights)
                    report=model.update([rows[int(i)] for i in selection],args.goal)
                    losses.append(report['loss'])
                save_atomic(ckpt,model,meta,e+1)
                result=dict(stage='epoch_committed',epoch=e+1,updates=model.updates,
                            mean_loss=round(float(np.mean(losses)),6),
                            real_successful_action_outcomes=stats['actual_successful_action_outcomes'],
                            source_committed_cursor=stats['committed_cursor'],
                            wall_s=round(time.perf_counter()-started,3))
                log.write(json.dumps(result,sort_keys=True)+'\n');log.flush()
                print('REPLAY Q EPOCH',json.dumps(result,sort_keys=True),flush=True)
            print('REPLAY Q SAVED',str(ckpt),flush=True)

if __name__=='__main__':main()
