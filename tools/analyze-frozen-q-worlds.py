#!/usr/bin/env python3
"""Audit real frozen-Q Minecraft episode journals, not training-world anecdotes.

Reports actual acquired inventory, mined wood, visible-target engagement,
and matched-seed comparisons. Partial journals are flagged, never promoted
to completed heldout evidence.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read_tag(tag):
    path=ROOT/'logs'/f'fresh_eval_{tag}.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    config=next(x for x in rows if x.get('stage')=='config')
    if config.get('mode')!='eval' or not config.get('restricted_offpolicy_skill_evaluation'):
        raise ValueError(f'{tag}: journal is not a frozen Q-only evaluation')
    episodes=[e for e in rows if e.get('stage')=='episode_committed' and e.get('mode')=='eval']
    last=rows[-1]
    done=last.get('stage')=='completed' and last.get('new_optimizer_updates_in_eval')==0
    if done and (len(episodes)!=last.get('evaluated_episodes') or
                 last.get('seed_namespace')!='eval'):
        raise ValueError(f'{tag}: completed frozen evaluation failed consistency check')
    if len({e['episode'] for e in episodes})!=len(episodes):
        raise ValueError(f'{tag}: duplicated evaluation episode indices')
    return dict(tag=tag,config=config,episodes=episodes,done=done,last=last)


def stats(episodes):
    cnt=lambda name:sum(e['counts'].get(name,0) for e in episodes)
    audit=lambda name:sum(e.get('action_audit',{}).get(name,0) for e in episodes)
    return dict(worlds=len(episodes),positive_worlds=sum(e['counts'].get('wood_inventory_gain',0)>0 for e in episodes),
                wood_inventory_gains=cnt('wood_inventory_gain'),
                mined_wood_blocks=cnt('new_wood_mined'),
                mined_all_blocks=cnt('new_mined_blocks'),
                wood_sightings=cnt('new_visible_wood'),
                attacks_on_wood_crosshair=audit('attack_on_wood_decisions'),
                crosshair_wood_decisions=audit('crosshair_wood_decisions'),
                deaths=cnt('death'))


def report(models,*,partial=False):
    root_seed=None
    for m in models:
        cfg=m['config']
        if not m['done'] and not partial:raise ValueError(
            f"{m['tag']}: evaluation is still running or incomplete (use --allow-partial)")
        seed_root=cfg['metadata']['seed_root']
        if root_seed is None:root_seed=seed_root
        if seed_root!=root_seed:raise ValueError('Different frozen seed root; not comparable')
        st=stats(m['episodes'])
        print('MODEL',m['tag'],'source',cfg['q_model'],'RNG',cfg['metadata']['policy_seed'],
              'complete',m['done'],'worlds',st['worlds'],'positive_worlds',st['positive_worlds'],
              'wood_inventory_gains',st['wood_inventory_gains'],
              'wood_mined',st['mined_wood_blocks'],
              'wood_crosshair_attack',st['attacks_on_wood_crosshair'],
              'wood_sightings',st['wood_sightings'],'deaths',st['deaths'])
        for e in m['episodes']:
            c=e['counts']
            print('  SEED',e['episode'],e['seed'],'wood_gained',c.get('wood_inventory_gain',0),
                  'wood_mined',c.get('new_wood_mined',0),
                  'crosshair_wood',e.get('action_audit',{}).get('crosshair_wood_decisions',0))
    by_rng=defaultdict(list)
    for m in models:
        by_rng[m['config']['metadata']['policy_seed']].append(m)
    for rng,group in by_rng.items():
        if len(group)<2:continue
        reference=[(e['episode'],e['seed']) for e in group[0]['episodes']]
        for m in group[1:]:
            other=[(e['episode'],e['seed']) for e in m['episodes']]
            n=min(len(reference),len(other))
            if reference[:n]!=other[:n]:
                raise ValueError('Matched seed indices mismatch: '+m['tag'])
        print('MATCHED SEED RNG',rng,'models',','.join(m['tag'] for m in group),
              'overlap',min(len(m['episodes']) for m in group),
              '(separate real game runs, not identical action trajectories)')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tags',nargs='+',help='Frozen evaluation run tags')
    ap.add_argument('--allow-partial',action='store_true')
    args=ap.parse_args()
    report([read_tag(tag) for tag in args.tags],partial=args.allow_partial)


if __name__=='__main__':
    main()
