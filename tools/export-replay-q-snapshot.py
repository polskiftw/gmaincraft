#!/usr/bin/env python3
"""Freeze a COMMITTED replay-Q self-play collector for independent held-out tests.

The live source stays intact. Export is an exclusive new immutable checkpoint
under checkpoints/replay_q_<TAG>.pt and will NEVER silently overwrite one.
Uses only already committed network weights; no game data/eval leakage.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.fresh_seed_schedule import SCHEMA as STREAM_SCHEMA
from gmaincraft.natural_online_policy import TEMPORAL_V9_SCHEMA,MOTOR_V7_HEAD_SIZES
from gmaincraft.replay_q_skill import SCHEMA as Q_SCHEMA
from gmaincraft.replay_q_rare_v2 import SCHEMA as Q_V2_SCHEMA,WOOD_CRACK_GOAL


def export_snapshot(root,collector,tag):
    import torch
    for x in (collector,tag):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,22}',x):
            raise ValueError('Unsafe source or output Q tag')
    source=root/'checkpoints'/f'fresh_stream_{collector}.pt'
    journal=root/'logs'/f'fresh_stream_{collector}.jsonl'
    target=root/'checkpoints'/f'replay_q_{tag}.pt'
    if target.exists() or target.with_suffix('.pt.pending').exists():
        raise FileExistsError('Refusing to replace an existing or interrupted Q snapshot')
    original=torch.load(source,map_location='cpu',weights_only=True)
    meta=original['metadata']
    if (original.get('stream')!=STREAM_SCHEMA or original.get('schema') not in (Q_SCHEMA,Q_V2_SCHEMA) or
        meta.get('kind') not in ('gmaincraft_q_collect_v1','gmaincraft_q_collect_v2_rare_outcome') or
        meta.get('reward_profile')!='wood_temporal_v9' or
        meta.get('policy_schema')!=TEMPORAL_V9_SCHEMA or
        tuple(meta.get('policy_action_heads',()))!=MOTOR_V7_HEAD_SIZES or
        original.get('next_episode',0)<=meta.get('collector_start_index',0) or
        original.get('decisions',0)<=0 or
        original['model'].get('schema')!=original.get('schema') or
        (original.get('schema')==Q_V2_SCHEMA and meta.get('collector_q_goal')!=WOOD_CRACK_GOAL)):
        raise ValueError('Source is not a committed compatible real Q self-play collector')
    # Confirm exact last committed save position; incomplete next world's
    # journal rows may follow it but cannot become training evidence.
    entries=[json.loads(x) for x in journal.read_text().splitlines() if x.strip()]
    completed=[x for x in entries if x.get('stage')=='episode_committed']
    if (not completed or completed[-1].get('next_episode')!=original['next_episode'] or
        completed[-1].get('committed_decisions')!=original['decisions']):
        raise ValueError('Source Q checkpoint and real completed episode journal disagree')
    outmeta=dict(source_run=collector,source_observation_schema=TEMPORAL_V9_SCHEMA,
                 action_sizes=MOTOR_V7_HEAD_SIZES,goal=meta['collector_q_goal'],
                 source_seed_root=meta['seed_root'],
                 source_collected_worlds=len(completed),
                 source_collected_game_ticks=original['decisions']*6,
                 source_q_optimizer_updates=original['model']['updates'],
                 source_starting_q_model=meta['collector_q_model'],
                 provenance='copy_of_committed_self_play_q_checkpoint')
    if original['schema']==Q_V2_SCHEMA:
        # A V2 collection is an extension of PREVIOUS self-play. Never lose
        # its old genuine positive outcomes when freezing/restarting a newer
        # V2 collector. Preserve the original committed source identifier,
        # plus the newly frozen collector's own subsequent self-play.
        original_frozen=root/'checkpoints'/f"replay_q_{meta['collector_q_model']}.pt"
        prior=torch.load(original_frozen,map_location='cpu',weights_only=True)
        if prior.get('schema')!=Q_V2_SCHEMA:
            raise ValueError('A V2 collector must originate from an actual V2 self-play network')
        prior_meta=prior['metadata']
        ancestry=prior_meta.get('protected_prior_source_runs')
        if ancestry is None:
            ancestor=prior_meta.get('protected_prior_source_run',prior_meta.get('source_run'))
            ancestry=(ancestor,)
        ancestry=tuple(ancestry)
        immediate=prior_meta.get('source_run')
        lineage=ancestry+((immediate,) if immediate not in ancestry else ())
        if (not lineage or len(lineage)>128 or len(lineage)!=len(set(lineage)) or
            not all(isinstance(name,str) and
                        re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,22}',name)
                        for name in lineage) or
            prior_meta.get('source_seed_root')!=meta['seed_root'] or collector in lineage):
            raise ValueError('Unverifiable earlier V2 history / seed provenance')
        outmeta['protected_prior_source_runs']=lineage
        outmeta['protected_prior_source_run']=lineage[0]  # compatibility
    output=dict(schema=original['schema'],metadata=outmeta,model=original['model'],
                source_committed_cursor=original['next_episode'])
    staged=target.with_suffix('.pt.pending')
    with staged.open('xb') as handle:
        torch.save(output,handle)
        handle.flush();os.fsync(handle.fileno())
    # Atomic hardlink publication, exclusive even if another agent exports.
    os.link(staged,target,follow_symlinks=False)
    staged.unlink()
    fd=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(export=str(target),collector=collector,committed_worlds=len(completed),
                player_actions=completed[-1]['committed_decisions'],
                q_updates=original['model']['updates'],goal=meta['collector_q_goal'])


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('collector_run')
    ap.add_argument('new_snapshot_tag')
    args=ap.parse_args(argv)
    print('REPLAY Q SNAPSHOT',json.dumps(export_snapshot(ROOT,args.collector_run,
                                                       args.new_snapshot_tag),sort_keys=True))

if __name__=='__main__':main()
