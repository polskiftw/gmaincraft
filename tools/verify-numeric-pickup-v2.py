#!/usr/bin/env python3
"""Validate real Minecraft pickup results independently of Gamescope teardown.

A SIGSEGV in the *outer compositor* is not a model failure if the child
already completed and committed a valid journal+checkpoint. Never erase a
real Python exception, unfinished journal, optimizer corruption, or bad mode.
"""
import argparse
import fcntl
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def paths(tag):
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,22}', tag) is None:
        raise ValueError('Unsafe run tag')
    return (ROOT/'logs'/f'numeric_pickup_{tag}.jsonl',
            ROOT/'checkpoints'/f'numeric_pickup_{tag}.pt',
            ROOT/'checkpoints'/f'numeric_pickup_{tag}.lock')


def inspect(tag, *, target=None, require_complete=False, mode=None,
            source=None, verbose=True):
    journal, checkpoint, lock = paths(tag)
    if not journal.is_file():
        if require_complete:raise ValueError(f'Missing journal: {journal}')
        if verbose:print(f'No journal yet: {journal}')
        return dict(tag=tag,stage='not_started',episodes=0,active=False)
    try:
        rows=[json.loads(line) for line in journal.read_text().splitlines()
              if line.strip()]
    except (OSError,json.JSONDecodeError) as exc:
        raise ValueError(f'Corrupt or partial journal {tag}: {exc}') from exc
    if not rows or rows[0].get('stage')!='config':
        raise ValueError('Journal lacks run configuration')
    cfg=rows[0]
    if cfg.get('metadata',{}).get('schema')!='numeric_pickup_egocentric_replay_q_v2':
        raise ValueError('Expected versioned numeric pickup replay Q kit')
    if mode and cfg.get('mode')!=mode:
        raise ValueError(f'Wrong run mode: expected {mode}')
    if source is not None and cfg.get('frozen_source')!=source:
        raise ValueError(f'Wrong frozen source: expected {source}')
    attempts=[r for r in rows if r.get('stage')=='episode_committed']
    if any(r.get('episode')!=i for i,r in enumerate(attempts)):
        raise ValueError('Missing or duplicated committed episodes')
    if any(r.get('scored_controller_commands')!=0 or
           r.get('own_inventory_gain')!=int(r.get('actual_acquired')) or
           r.get('final_held_oak')!=r.get('own_inventory_gain') or
           r.get('own_inventory_gain') not in (0,1) or
           r.get('actor_updates',-1)<0 for r in attempts):
        raise ValueError('Contradictory scored action, inventory or update evidence')
    stage=rows[-1].get('stage')
    last=rows[-1]
    complete=stage=='completed'
    if stage=='paused' or complete:
        if (last.get('episodes')!=len(attempts) or
            last.get('successes')!=sum(int(r['actual_acquired']) for r in attempts) or
            last.get('inventory_items')!=sum(r['own_inventory_gain'] for r in attempts)):
            raise ValueError('Summary does not reconcile with actual inventory episodes')
    if require_complete and not complete:
        raise ValueError(f'Training incomplete: last journal stage {stage}, '
                         f'{len(attempts)}/{target} episodes')
    if target is not None and require_complete and len(attempts)!=target:
        raise ValueError(f'Wrong final count: {len(attempts)}, expected {target}')
    if lock.exists():
        with lock.open('a+') as held:
            try:
                fcntl.flock(held, fcntl.LOCK_EX|fcntl.LOCK_NB)
                active=False
                fcntl.flock(held, fcntl.LOCK_UN)
            except BlockingIOError:
                active=True
    else:
        active=False
    saved_updates=0
    if cfg.get('mode')=='train':
        if not checkpoint.is_file():raise ValueError('Training has no saved checkpoint')
        import torch
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        cursor=saved.get('next_episode')
        # A status request reads the journal BEFORE loading a potentially
        # multi-megabyte checkpoint. Real Minecraft may commit several
        # episodes in the meantime; allow checkpoint to be ahead ONLY while
        # that run holds the actual lock. Completed checks remain strict.
        live_snapshot=(active and not require_complete and
                       type(cursor) is int and len(attempts)<=cursor<=len(attempts)+100)
        if (saved.get('schema')!='numeric_pickup_egocentric_replay_q_v2' or
            json.loads(json.dumps(saved.get('metadata')))!=cfg['metadata'] or
            (cursor!=len(attempts) and not live_snapshot) or
            saved.get('brain',{}).get('episodes_seen')!=cursor):
            raise ValueError('Checkpoint cursor / schema does not match journal')
        saved_updates=saved['brain']['updates']
        if attempts and saved_updates!=attempts[-1].get('actor_updates'):
            if not (live_snapshot and saved_updates>=attempts[-1]['actor_updates']):
                raise ValueError('Training checkpoint optimizer not synced to journal')
        if complete and last['optimizer_updates']!=saved_updates:
            raise ValueError('Training summary optimizer updates disagree')
    elif cfg.get('mode')=='eval':
        if complete and last.get('optimizer_updates')!=0:
            raise ValueError('Evaluation illegally updated optimizer')
        if source is None:
            source=cfg.get('frozen_source')
        if not source:raise ValueError('Evaluation missing frozen model source')
        _,model_path,_=paths(source)
        if not model_path.is_file():raise ValueError('Missing frozen model checkpoint')
        import torch
        saved=torch.load(model_path,map_location='cpu',weights_only=True)
        saved_updates=saved['brain']['updates']
        if any(r['actor_updates']!=saved_updates for r in attempts):
            raise ValueError('Frozen model updated during evaluation')
    else:
        raise ValueError('Unexpected experiment mode')
    result=dict(tag=tag,mode=cfg['mode'],source=cfg.get('frozen_source'),
                stage=stage,active=active,episodes=len(attempts),
                successes=sum(int(x['actual_acquired']) for x in attempts),
                seen_at_start=sum(int(x['item_visible_at_first_action']) for x in attempts),
                expected=target,optimizer_updates=saved_updates,
                **({'checkpoint_cursor':cursor} if cfg['mode']=='train' else {}))
    if verbose:
        print('NUMERIC PICKUP VERIFIED',json.dumps(result,sort_keys=True),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('tag')
    p.add_argument('--target',type=int)
    p.add_argument('--require-complete',action='store_true')
    p.add_argument('--mode',choices=('train','eval'))
    p.add_argument('--source')
    a=p.parse_args()
    try:
        inspect(a.tag,target=a.target,require_complete=a.require_complete,
                mode=a.mode,source=a.source)
    except (ValueError,OSError,KeyError) as exc:
        p.exit(1,f'INVALID NUMERIC PICKUP RUN: {exc}\n')

if __name__=='__main__':main()
