#!/usr/bin/env python3
"""Audit complete frozen numeric-memory item-pickup evaluations.

Requires same staged eval seed, identical case placement/yaw per episode,
same action budgets and immutable checkpoint ancestry. Never promote partial
journals to a learned skill. The staged item itself is NOT a natural-world win.
"""
import argparse
import json
from pathlib import Path
from math import sqrt

ROOT=Path(__file__).resolve().parents[1]


def read_run(tag):
    path=ROOT/'logs'/f'numeric_pickup_{tag}.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    config=rows[0]
    eps=[x for x in rows if x.get('stage')=='episode_committed']
    final=rows[-1]
    if (config.get('stage')!='config' or config.get('mode')!='eval'
        or final.get('stage')!='completed' or
        final.get('optimizer_updates')!=0 or len(eps)!=final.get('episodes')
        or not config.get('frozen_source')):
        raise ValueError(f'{tag}: not a complete immutable-policy evaluation')
    if any(x['episode']!=i for i,x in enumerate(eps)):
        raise ValueError(f'{tag}: nonconsecutive/duplicated episodes')
    return dict(tag=tag,config=config,episodes=eps,final=final)


def wilson(success,n):
    z=1.96
    mean=success/n
    center=(mean+z*z/(2*n))/(1+z*z/n)
    half=z*sqrt(mean*(1-mean)/n+z*z/(4*n*n))/(1+z*z/n)
    return [round(max(0.,center-half),3),round(min(1.,center+half),3)]


def report(runs):
    ref=runs[0]
    for r in runs[1:]:
        if r['config']['metadata']!=ref['config']['metadata']:
            raise ValueError('Incompatible numeric item observation/action/case source')
        cases=lambda k:[(x['case_seed'],x['case_dx'],x['case_dz'],x['case_yaw'])
                        for x in k['episodes']]
        if cases(r)!=cases(ref):
            raise ValueError('Different controller-staged item positions: not matched')
    for r in runs:
        es=r['episodes']; n=len(es)
        wins=sum(x['actual_acquired'] for x in es)
        if any(x['scored_controller_commands']!=0 for x in es):
            raise ValueError('Privileged scored action detected')
        print('FROZEN',r['tag'],'source',r['config']['frozen_source'],
              'n',n,'actual_pickups',wins,'rate',round(wins/n,3),
              '95%_Wilson',wilson(wins,n),
              'outside_boundary',sum(x.get('practice_out_of_bounds',False) for x in es),
              'mean_normal_actions',round(sum(x['decisions'] for x in es)/n,2),
              'mean_visible_steps',round(sum(x['visible_decisions'] for x in es)/n,2))
    if len(runs)>1:
        print('PAIRED_CASES',len(ref['episodes']),
              'NOTE independent action trajectories, not causal significance proof')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tags',nargs='+')
    args=ap.parse_args()
    report([read_run(tag) for tag in args.tags])


if __name__=='__main__':main()
