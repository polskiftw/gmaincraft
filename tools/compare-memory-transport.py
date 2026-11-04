#!/usr/bin/env python3
"""Compare frozen same-model Minecraft RGB versus opt-in zero-RGB observations.

Requires COMPLETE journals, EXACT eval seed index+seed, policy seed, model,
goal, reward profile and action horizon. Never compare partially committed
episodes or claim a visibility improvement from a faster transport.
"""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def load(tag):
    rows=[json.loads(x) for x in (ROOT/'logs'/f'fresh_eval_{tag}.jsonl').read_text().splitlines()]
    config=next(e for e in rows if e.get('stage')=='config')
    ep=[e for e in rows if e.get('stage')=='episode_committed']
    final=rows[-1]
    if final.get('stage')!='completed' or final.get('new_optimizer_updates_in_eval')!=0:
        raise ValueError(tag+': frozen evaluation not completed cleanly')
    if len(ep)!=final.get('evaluated_episodes'):
        raise ValueError(tag+': unexpected number of episodes')
    return config,ep,final


def compare(rgb,without):
    a,e1,s1=load(rgb)
    b,e2,s2=load(without)
    keep=('q_model','q_goal','evaluation_control')
    for key in keep:
        if a.get(key)!=b.get(key):
            raise ValueError('Model/goal/control mismatch at '+key)
    ma=dict(a['metadata'])
    mb=dict(b['metadata'])
    transport=mb.pop('frame_transport',None)
    if transport!='memory_only_frame_callback_v1' or 'frame_transport' in ma or ma!=mb:
        raise ValueError('Not a matched default-RGB vs experimental zero-RGB metadata pair')
    keys=lambda rows:[(e['episode'],e['seed'],e['game_ticks']) for e in rows]
    if keys(e1)!=keys(e2):
        raise ValueError('Frozen worlds/game tick budgets do not match')
    def view(name,eps,summary):
        ticks=sum(e['game_ticks'] for e in eps)
        io_s=sum(e['game_io_s'] for e in eps)
        owned=sum(e['counts'].get('wood_inventory_gain',0) for e in eps)
        print(name,'worlds',len(eps),'ticks',ticks,'mean_game_io_ms_per_tick',
              round(io_s*1000/ticks,4),'summary_wall_seconds',summary['wall_s'],
              'wall_ms_per_game_tick',round(summary['wall_s']*1000/ticks,4),
              'wood_inventory_gains',owned)
        return io_s,ticks
    print('MATCHED frozen model',a['q_model'],'goal',a['q_goal'],'seed_root',ma['seed_root'],
          'action_seed',ma['policy_seed'])
    old,ticks=view('RGB',e1,s1)
    new,_=view('ZERO_RGB',e2,s2)
    print('Game IO time reduction',round(100*(old-new)/old,2),'percent')
    print('NOTE both modes still render the game; this compares readback + packet transport')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rgb_eval_tag')
    parser.add_argument('no_rgb_eval_tag')
    args=parser.parse_args()
    compare(args.rgb_eval_tag,args.no_rgb_eval_tag)
