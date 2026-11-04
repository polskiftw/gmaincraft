"""Frozen Q policy action-probabilities on SAME committed player-only MCio states.

Not an online success metric: sampling the *saved real observations* permits
apples-to-apples diagnostics of what changed in the motor, without changing
Minecraft worlds, installing preaiming scripts, or revealing hidden state.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.experience_archive import FIELDS,archive_path,read_episode
from gmaincraft.natural_online_policy import MOTOR_V7,TEMPORAL_V9_SELF_FEATURES
from gmaincraft.replay_q_skill import ReplayQSkill,CRACK_GOAL,SCHEMA as Q_SCHEMA
from gmaincraft.replay_q_rare_v2 import RareOutcomeQSkill,SCHEMA as Q_V2_SCHEMA

WOOD=TEMPORAL_V9_SELF_FEATURES.index('crosshair_is_wood')
SAME=TEMPORAL_V9_SELF_FEATURES.index('same_vanilla_crosshair_target_as_previous')
ATTACK_INDEX=[i for i,(_,_,_,attack,*rest) in enumerate(MOTOR_V7) if attack]
STILL_INDEX=[i for i,entry in enumerate(MOTOR_V7) if entry[0].startswith('attack_stationary')]
LONG_INDEX=[i for i,entry in enumerate(MOTOR_V7) if entry[-1]>=24]


def load_own_states(run,episode_indices,limit=2000):
    all_rows=[]
    for ep in episode_indices:
        path=archive_path(ROOT/'data-local'/'experience',run,ep)
        meta,data=read_episode(path)
        n=meta['count']
        if n>limit:
            order=np.linspace(0,n-1,num=limit,dtype=int)
        else:order=np.arange(n)
        for ix in order:
            all_rows.append({('before_'+f):data['before_'+f][ix] for f in FIELDS})
    return all_rows


def frozen_probs(model,rows,*,goal=CRACK_GOAL,batch_size=128):
    outputs=[]
    model.net.eval()
    with model.torch.no_grad():
        for index in range(0,len(rows),batch_size):
            batch=rows[index:index+batch_size]
            q=model.net(*model._batch(batch,'before_',goal))
            p=model.torch.softmax(q/.2,dim=-1)
            outputs.append(p.detach().cpu().numpy())
    return np.concatenate(outputs,axis=0)


def audit(rows,probs):
    own=np.stack([x['before_own'] for x in rows]);mask=own[:,WOOD]>.5
    result=dict(actual_observations=len(rows),wood_crosshair_actual_states=int(mask.sum()))
    for group,selected in (('all',np.ones(len(rows),dtype=bool)),
                           ('wood_under_crosshair',mask),
                           ('no_wood_under_crosshair',~mask)):
        if not selected.any():continue
        p=probs[selected]
        result[group]=dict(samples=len(p),attack_prob=round(float(p[:,ATTACK_INDEX].sum(axis=1).mean()),6),
                           stationary_attack_prob=round(float(p[:,STILL_INDEX].sum(axis=1).mean()),6),
                           long_action_prob=round(float(p[:,LONG_INDEX].sum(axis=1).mean()),6),
                           greedy_attack_fraction=round(float(np.isin(p.argmax(axis=1),ATTACK_INDEX).mean()),6),
                           greedy_still_fraction=round(float(np.isin(p.argmax(axis=1),STILL_INDEX).mean()),6))
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-run',default='qselfplay_v1_1104')
    ap.add_argument('--episodes',default='2,5,15,36,100,199')
    ap.add_argument('--models',default='qcrack_v1a1104,qselfplay_final1104')
    ap.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=ap.parse_args(argv)
    indices=[int(x) for x in args.episodes.split(',')]
    rows=load_own_states(args.source_run,indices)
    import torch
    for name in args.models.split(','):
        record=torch.load(ROOT/'checkpoints'/f'replay_q_{name}.pt',map_location='cpu',weights_only=True)
        if record.get('schema') not in (Q_SCHEMA,Q_V2_SCHEMA):raise ValueError('Unexpected archived Q checkpoint')
        goal=record['metadata']['goal']
        model=(RareOutcomeQSkill if record['schema']==Q_V2_SCHEMA else ReplayQSkill)(device=args.device)
        model.restore(record['model'],optimizer=False)
        probs=frozen_probs(model,rows,goal=goal)
        print('REPLAY Q AUDIT',json.dumps(dict(model=name,updates=model.updates,
              source_run=args.source_run,episode_indices=indices,goal=goal,
              sensor_based_original_observation_only=True,
              frozen_offline_not_gameplay_success=True,**audit(rows,probs)),sort_keys=True))

if __name__=='__main__':main()
