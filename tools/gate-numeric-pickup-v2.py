#!/usr/bin/env python3
"""Strict paired, multiple-seed frozen pickup promotion gate (no score hacking).

For every CASE SEED require complete frozen journals from greedy or sampled,
uniform random and always-forward, exactly matching staged episode cases.
The policy must exceed BOTH controls on pooled real inventory pickups by
at least 10 percentage points. Training curves/rewards never qualify.
"""
import argparse,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def load(tag):
    if not tag.isascii() or not tag.replace('_','').replace('-','').isalnum():
        raise ValueError('Invalid tag')
    path=ROOT/'logs'/f'numeric_pickup_{tag}.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or rows[0].get('stage')!='config' or rows[0].get('mode')!='eval':
        raise ValueError(f'{tag}: not a scored frozen evaluation')
    config=rows[0]
    episodes=[x for x in rows if x.get('stage')=='episode_committed']
    if (not episodes or rows[-1].get('stage')!='completed' or
        rows[-1].get('optimizer_updates')!=0 or
        len(episodes)!=rows[-1].get('episodes') or
        any(row.get('episode')!=i for i,row in enumerate(episodes)) or
        any(row.get('scored_controller_commands')!=0 or
            row.get('own_inventory_gain',0)<int(row['actual_acquired'])
            for row in episodes)):
        raise ValueError(f'{tag}: incomplete / untrusted frozen episode journal')
    return config,episodes


def gate(groups, *, threshold=.10):
    if len(groups)<2:
        raise ValueError('At least two independently seeded heldout suites required')
    all_counts=[0,0,0]
    total=0
    seeds=set()
    model=None
    for policy_tag,random_tag,forward_tag in groups:
        three=[load(t) for t in (policy_tag,random_tag,forward_tag)]
        configs=[r[0] for r in three]
        eps=[r[1] for r in three]
        seed=configs[0]['metadata']['case_seed']
        if seed in seeds:raise ValueError('Heldout seed reused')
        seeds.add(seed)
        if (any(c['metadata']!=configs[0]['metadata'] for c in configs[1:]) or
            any(c.get('frozen_source')!=configs[0].get('frozen_source') for c in configs[1:]) or
            any(len(e)!=len(eps[0]) for e in eps)):
            raise ValueError('Mismatched checkpoint ancestry, contract or case count')
        if configs[0]['metadata'].get('schema')!='numeric_pickup_egocentric_replay_q_v2':
            raise ValueError('Not the new kit')
        if model is not None and model!=configs[0].get('frozen_source'):
            raise ValueError('Different frozen model between suites')
        model=configs[0]['frozen_source']
        keys=[[(x['case_dx'],x['case_dz'],x['case_yaw'],x['case_seed']) for x in e]
              for e in eps]
        if keys[0]!=keys[1] or keys[0]!=keys[2]:
            raise ValueError('Unpaired scenario positions')
        counts=[sum(bool(x['actual_acquired']) for x in e) for e in eps]
        total+=len(eps[0]);all_counts=[a+b for a,b in zip(all_counts,counts)]
        print(f'CASE_SEED {seed} n={len(eps[0])} learned={counts[0]} random={counts[1]} forward={counts[2]}')
    rates=[v/total for v in all_counts]
    margin=rates[0]-max(rates[1:])
    passed=(total>=200 and margin>=threshold)
    print('GATE', 'PASS_PROVISIONAL_PRACTICE' if passed else 'FAIL_NO_PROMOTION',
          'model',model,'n',total,'real_inventory_pickups',all_counts,
          'rates',[round(x,4) for x in rates],
          'margin_vs_strongest_control',round(margin,4),
          'required_margin',threshold,
          'NOTE: One network, staged pickup, not natural-world mastery')
    return passed


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('tags',nargs='+',help='Triples: learned random always-forward (repeat for each seed)')
    a=p.parse_args()
    if len(a.tags)%3: p.error('Supply 3 tags per independent seed')
    try:
        success=gate([a.tags[i:i+3] for i in range(0,len(a.tags),3)])
    except (ValueError,KeyError,FileNotFoundError) as exc:
        p.exit(2,f'INVALID EVALUATION: {exc}\n')
    sys.exit(0 if success else 1)
