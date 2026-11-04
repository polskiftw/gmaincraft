"""Frozen V9 response audit on actual OWN saved observations; no Minecraft launch.

Companion to unbiased game-world evaluations. An artificial ablation sets
only the V9 HISTORY channels to zero in already legitimately recorded
observations. It is a diagnostic perturbation, NEVER a real game rollout,
and it does not establish a learned skill or a causal performance benefit.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.experience_archive import read_episode
from gmaincraft.natural_online_policy import (
    NaturalObservation,NaturalPPO,TEMPORAL_V9_SCHEMA,TEMPORAL_V9_SELF_FEATURES,
    TEMPORAL_V9_FEATURES,MOTOR_V7,MOTOR_V7_HEAD_SIZES)

WOOD_FIELD=TEMPORAL_V9_SELF_FEATURES.index('crosshair_is_wood')
CONTINUITY_FIELD=TEMPORAL_V9_SELF_FEATURES.index('same_vanilla_crosshair_target_as_previous')
ATTACK_STATIONARY=tuple(i for i,row in enumerate(MOTOR_V7) if row[0].startswith('attack_stationary'))


def summarize_metrics(probabilities,ablated,features):
    """Input is actual player-known states and frozen inferred choices."""
    if (probabilities.shape!=ablated.shape or probabilities.ndim!=2 or
        probabilities.shape[1]!=len(MOTOR_V7) or features.ndim!=2 or
        features.shape!=(len(probabilities),len(TEMPORAL_V9_SELF_FEATURES)) or
        not np.isfinite(probabilities).all() or not np.isfinite(ablated).all()):
        raise ValueError('Invalid actual V9 policy-response samples')
    wood=features[:,WOOD_FIELD]>.5
    same=features[:,CONTINUITY_FIELD]>.5
    p=probabilities[:,ATTACK_STATIONARY].sum(axis=1)
    p0=ablated[:,ATTACK_STATIONARY].sum(axis=1)
    distribution_change=np.abs(probabilities-ablated).sum(axis=1)/2
    def mean(a):return round(float(a.mean()),6) if len(a) else None
    return dict(observed_decisions=len(probabilities),
                actual_wood_crosshair_decisions=int(wood.sum()),
                actual_previous_same_target_decisions=int(same.sum()),
                mean_stationary_attack_probability=mean(p),
                mean_stationary_attack_probability_on_actual_wood=mean(p[wood]),
                mean_stationary_attack_probability_on_other_observations=mean(p[~wood]),
                mean_stationary_attack_probability_with_history_zeroed=mean(p0),
                mean_absolute_stationary_attack_delta_from_history_ablation=mean(np.abs(p-p0)),
                mean_total_variation_from_history_ablation=mean(distribution_change),
                max_total_variation_from_history_ablation=round(float(distribution_change.max()),6))


def run(run_tag,max_episodes,device):
    import torch
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,22}',run_tag):
        raise ValueError('Unsafe train tag')
    checkpoint=ROOT/'checkpoints'/f'fresh_stream_{run_tag}.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if (saved['metadata']['policy_schema']!=TEMPORAL_V9_SCHEMA or
        saved['metadata']['reward_profile']!='wood_temporal_v9'):
        raise ValueError('Requires a native V9 learned-target-history training checkpoint')
    model=NaturalPPO(device=device,schema=TEMPORAL_V9_SCHEMA,
                     action_sizes=MOTOR_V7_HEAD_SIZES)
    model.restore(saved['model'],optimizer=False)
    archive=ROOT/'data-local'/'experience'/run_tag
    paths=sorted(p for p in archive.glob('episode_*.npz') if
                 re.fullmatch(r'episode_[0-9]{12}\.npz',p.name) and
                 not p.is_symlink() and int(p.name[8:20])<saved['next_episode'])[-max_episodes:]
    if not paths:
        raise ValueError('No checkpoint-committed V9 player-only recorded experience')
    observed=[]
    for file in paths:
        meta,data=read_episode(file)
        if meta['observation_schema']!=TEMPORAL_V9_SCHEMA:
            raise ValueError('Archive observation schema differs from frozen model')
        for k in range(meta['count']):
            observed.append(NaturalObservation(*(data['before_'+name][k] for name in
                              ('own','crosshair_id','surface_ids','surface_data','surface_mask'))))
    original=[]
    no_history=[]
    model.net.eval()
    with torch.no_grad():
        for first in range(0,len(observed),128):
            batch=observed[first:first+128]
            logits,_=model.net(*model._batch(batch))
            original.extend(torch.softmax(logits[0],dim=-1).cpu().numpy())
            ablated=[]
            for obs in batch:
                own=obs.own.copy()
                own[-len(TEMPORAL_V9_FEATURES):]=0
                ablated.append(replace(obs,own=own))
            logits0,_=model.net(*model._batch(ablated))
            no_history.extend(torch.softmax(logits0[0],dim=-1).cpu().numpy())
    result=summarize_metrics(np.array(original),np.array(no_history),
                             np.stack([obs.own for obs in observed]))
    return dict(run_tag=run_tag,model_ppo_updates=saved['model']['updates'],
                source_archived_episodes=len(paths),
                note='Actual own observations, frozen network; zeroed history is ONLY an offline ablation, not real gameplay',
                **result)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_tag')
    p.add_argument('--max-episodes',type=int,default=30)
    p.add_argument('--device',choices=('cuda','cpu','auto'),default='auto')
    a=p.parse_args(argv)
    if not 1<=a.max_episodes<=100000: p.error('Invalid number of episodes')
    print(json.dumps(run(a.run_tag,a.max_episodes,a.device),sort_keys=True,indent=2))


if __name__=='__main__':main()
