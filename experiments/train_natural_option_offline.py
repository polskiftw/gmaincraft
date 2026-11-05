#!/usr/bin/env python3
"""Offline, conservative double-Q OPTION learner from COMMITTED real survival.

No teacher/mining solution/imagined success. The loader accepts ONLY opted-in
real own-state/action/outcome traces from completed frozen natural Minecraft
noise-world collections in the TRAIN seed namespace; heldout EVAL streams
are forbidden as training sources. It refuses all-zero outcomes or one-option
activity datasets rather than publishing junk weights. Not a speedrunner.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.fresh_seed_schedule import SeedSchedule
from gmaincraft.natural_option_q_v1 import NaturalOptionQ,validate_real_row,SCHEMA


def load_committed_real_trace(tag):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,22}',tag):
        raise ValueError('Unsafe committed source tag')
    path=ROOT/'logs'/f'fresh_eval_{tag}.jsonl'
    rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    if not rows or rows[0].get('stage')!='config' or rows[-1].get('stage')!='completed':
        raise ValueError(f'Incomplete actual Minecraft option replay: {tag}')
    cfg,last=rows[0],rows[-1]
    meta=cfg['metadata']
    if (cfg.get('mode')!='eval' or last.get('mode')!='eval' or
        cfg.get('evaluation_control')!='q_sampled' or
        cfg.get('q_goal')!='effect:visible_wood_crack_progress_v2' or
        not cfg.get('q_model') or
        meta.get('record_real_option_experience') is not True or
        meta.get('option_trace_schema')!=SCHEMA or
        meta.get('option_data_namespace')!='train' or
        last.get('seed_namespace')!='train' or
        last.get('new_optimizer_updates_in_eval')!=0 or
        meta.get('gameplay_commands')!=0 or
        not meta.get('paired_world_cache') or
        meta.get('pickup_transfer_contract')!='player_seen_dropped_wood_pickup_transfer_budget_v2' or
        not meta.get('transfer_source_hash')):
        raise ValueError(f'Trainable replay must come from complete, frozen real TRAIN-noise worlds: {tag}')
    episodes=[x for x in rows if x.get('stage')=='episode_committed']
    if (len(episodes)!=last.get('evaluated_episodes') or
        sum(x.get('decisions',0) for x in episodes)!=last.get('decisions')):
        raise ValueError('Committed Minecraft episode cursor does not match journal')
    schedule=SeedSchedule(meta['seed_root'],'train')
    replay=[]
    outcome=Counter()
    for i,ep in enumerate(episodes):
        if (ep.get('episode')!=cfg['evaluation_seed_start']+i or
            ep.get('seed')!=schedule.seed_at(ep['episode']) or
            ep.get('updates')!=0 or
            ep.get('first_observation_hash') is None or
            ep.get('first_q_rng_hash') is None):
            raise ValueError('Option trace world is not a complete real TRAIN-noise seed')
        actions=ep.get('option_trace')
        if not isinstance(actions,list) or not actions:
            raise ValueError('Missing actual per-decision option trace')
        if len(actions)!=ep['policy_actions']:
            raise ValueError('Option trace omits real selected controller actions')
        for row in actions:
            validate_real_row(row)
            replay.append(row)
        if sum(row['ticks'] for row in actions)!=ep['game_ticks']:
            raise ValueError('Option observations do not span the real game-time budget')
        if sum(row['wood_inventory_gain'] for row in actions)!=ep['counts'].get('wood_inventory_gain',0):
            raise ValueError('Faked/unmatched own inventory outcome in option trace')
        if sum(row['wood_mined'] for row in actions)!=ep['counts'].get('new_wood_mined',0):
            raise ValueError('Faked/unmatched own block-mined outcome in option trace')
        if sum(row['option'] for row in actions)!=ep.get('pickup_transfer_audit',{}).get('pickup_policy_actions',0):
            raise ValueError('Pickup option choices disagree with actual game actions')
        outcome['worlds']+=1
        outcome['wood_mined']+=ep['counts'].get('new_wood_mined',0)
        outcome['wood_gained']+=ep['counts'].get('wood_inventory_gain',0)
    outcome['real_option_actions']=len(replay)
    outcome['pickup_option_actions']=sum(row['option'] for row in replay)
    outcome['pickup_gains_during_option']=sum(row['wood_inventory_gain'] for row in replay if row['option']==1)
    return replay,dict(outcome),dict(meta,
        source_q_model=cfg['q_model'],source_q_goal=cfg['q_goal'])


def gather(tags):
    rows=[];totals=Counter();sources=[];source_hash=None;model_lineage=None
    if not tags or len(tags)!=len(set(tags)):
        raise ValueError('Require unique real Minecraft source tags')
    for tag in tags:
        examples,counts,meta=load_committed_real_trace(tag)
        if source_hash is None:source_hash=meta['transfer_source_hash']
        if source_hash!=meta['transfer_source_hash']:
            raise ValueError('Do not merge different instrumented controller builds')
        lineage=(meta['source_q_model'],meta['source_q_goal'],meta['pickup_model'],
                 meta['pickup_transfer_contract'],meta['seed_root'])
        if model_lineage is None:model_lineage=lineage
        if lineage!=model_lineage:
            raise ValueError('Do not train from different frozen mining/pickup source models')
        for row in examples:rows.append(row)
        totals.update(counts)
        sources.append(tag)
    return rows,dict(totals),source_hash


def train(*,tag,sources,seed=20261105,steps=240,batch_size=64):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,22}',tag):
        raise ValueError('Unsafe new natural option run name')
    model_path=ROOT/'checkpoints'/f'natural_option_{tag}.pt'
    journal_path=ROOT/'logs'/f'natural_option_{tag}.train.json'
    if model_path.exists() or journal_path.exists():
        raise FileExistsError('Refusing to overwrite an existing natural option learner')
    examples,counts,source_hash=gather(sources)
    if (len(examples)<128 or counts['wood_gained']<2 or
        counts['pickup_option_actions']<8 or
        counts['pickup_gains_during_option']<1 or
        len(examples)-counts['pickup_option_actions']<8):
        raise ValueError(f'INSUFFICIENT REAL OUTCOMES, not training junk: {counts}')
    model=NaturalOptionQ(seed=seed)
    metrics=model.train_from_real_rows(examples,steps=steps,batch_size=batch_size)
    meta=dict(schema=SCHEMA,source_hash=source_hash,
              real_train_sources=tuple(sources),seed_namespace='train',
              training_outcomes=counts,optimizer_updates=model.updates,
              purpose='TRAINABLE option chooser / NOT natural-world generalization')
    record=dict(schema=SCHEMA,metadata=meta,learner=model.state())
    model_path.parent.mkdir(parents=True,exist_ok=True)
    journal_path.parent.mkdir(parents=True,exist_ok=True)
    temp=model_path.with_suffix('.pt.pending')
    if temp.exists():raise FileExistsError('Unfinished previous option model save; inspect')
    import torch
    with temp.open('xb') as out:
        torch.save(record,out)
        out.flush();os.fsync(out.fileno())
    os.replace(temp,model_path)
    with journal_path.open('x') as out:
        json.dump(dict(metadata=meta,metrics=metrics),out,indent=2,sort_keys=True)
        out.write('\n')
    print('NATURAL OPTION Q TRAINED ON REAL OUTCOMES',json.dumps(dict(tag=tag,
          sources=sources,counts=counts,metrics=metrics),sort_keys=True),flush=True)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag',required=True)
    p.add_argument('--source',action='append',required=True)
    p.add_argument('--seed',type=int,default=20261105)
    p.add_argument('--steps',type=int,default=240)
    p.add_argument('--batch-size',type=int,default=64)
    args=p.parse_args()
    try:train(tag=args.tag,sources=args.source,seed=args.seed,
              steps=args.steps,batch_size=args.batch_size)
    except (ValueError,KeyError,FileNotFoundError,FileExistsError) as exc:
        p.exit(1,'NATURAL OPTION DATA/LEARNING FAILED: '+str(exc)+'\n')

if __name__=='__main__':main()
