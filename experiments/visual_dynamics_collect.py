"""Collect experience from RANDOM normal Minecraft actions for vision learning.

Observation: two real consecutive RGB frames, not coordinates/block IDs.
Action labels: OUR OWN selected move/look/attack factors, not human demos.
Controller can reset a disposable real-log arena between episodes; only
standard Minecraft input operates during each episode. No trained model or
external data, no pretrained weights, no scripted speedrun planner.

Only pixel-history/actions/terminal flags are persisted as validated NPZ.
Never mix held-out --split validation into training of the representation.
"""
import argparse
import json
from pathlib import Path
import random
import re
import sys
from collections import Counter
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.agent_interface import MCioAgentBridge,AgentAction,natural_log_count
from gmaincraft.joint_curriculum import sample_near_cases
from gmaincraft.vision_ppo import PixelHistory
from gmaincraft.vision_factorized import (ACTION_TICKS,action_schema,HEAD_SIZES,
                                           FactorTransition,action_for_factors)
from gmaincraft.rollout_store import save_rollout_batch
from joint_log_ppo import launch_world


def parse_args(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--split',required=True,choices=('train','validation'))
    ap.add_argument('--scenario-seed',type=int,required=True)
    ap.add_argument('--action-seed',type=int,required=True)
    ap.add_argument('--episodes',type=int,default=16)
    ap.add_argument('--decisions',type=int,default=40)
    ap.add_argument('--action-regime',choices=('joint','movement_only'),default='joint')
    ap.add_argument('--ticks-per-decision',type=int,choices=(6,12),default=6)
    ap.add_argument('--frame-width',type=int,default=320)
    ap.add_argument('--frame-height',type=int,default=180)
    ap.add_argument('--visible',action='store_true')
    args=ap.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',args.run_tag):ap.error('Invalid run-tag')
    if args.episodes<1 or args.decisions<2 or args.episodes>5000 or args.decisions>200:
        ap.error('Invalid episodes/decisions')
    if not 96<=min(args.frame_width,args.frame_height) and max(args.frame_width,args.frame_height)<=1920:
        ap.error('Invalid frame size')
    return args


def main(argv=None):
    args=parse_args(argv)
    local=ROOT/'runs'/f'visual_dynamics_{args.run_tag}'
    world=ROOT/'mcio-data'/'instances'/'DemoInstance'/'saves'/f'MDL_VisualDynamics_{args.run_tag}'
    manifest=ROOT/'logs'/f'visual_dynamics_{args.run_tag}.jsonl'
    if local.exists() or world.exists() or manifest.exists():
        raise FileExistsError('Run tag already exists; refusing overwrite')
    rng=random.Random(args.action_seed)
    cases=sample_near_cases(args.scenario_seed,args.episodes)
    local.mkdir(parents=True,exist_ok=False)
    manifest.parent.mkdir(parents=True,exist_ok=True)
    print(f'VISUAL-DYNAMICS capture: {args.split} cases={len(cases)}, '
          f'decisions={args.decisions}, regime={args.action_regime}, '
          f'private outcome labels NOT stored in RGB data',flush=True)
    with manifest.open('x',encoding='utf-8') as manifest_file:
        manifest_file.write(json.dumps(dict(stage='config',split=args.split,
                              scenario_seed=args.scenario_seed,action_seed=args.action_seed,
                              episodes=args.episodes,decisions=args.decisions,
                              action_regime=args.action_regime,
                              ticks_per_decision=args.ticks_per_decision))+'\n')
        manifest_file.flush()
        def collect(arena):
            results=Counter()
            for i,case in enumerate(cases,1):
                packet=arena.reset(case)
                bridge=MCioAgentBridge(arena.client.ctrl,packet)
                frames=PixelHistory()
                image=frames.reset(bridge.view())
                steps=[]
                for t in range(args.decisions):
                    if args.action_regime=='movement_only':
                        factors=(rng.randrange(HEAD_SIZES[0]),0,0)
                        active_heads=(True,False,False)
                    else:
                        factors=tuple(rng.randrange(n) for n in HEAD_SIZES)
                        active_heads=(True,True,True)
                    proto=action_for_factors(factors)
                    action=AgentAction(keys=proto.keys,look_dx=proto.look_dx,
                                       look_dy=proto.look_dy,ticks=args.ticks_per_decision)
                    view=bridge.apply(action)
                    ended=(t==args.decisions-1 or natural_log_count(bridge.packet)>0
                           or bridge.packet.health<=0)
                    # Placeholders for diagnostic/unsupervised data, *not*
                    # policy estimates. Categorical policy chose uniformly.
                    log_prob=-sum(float(np.log(n)) for n,active in zip(HEAD_SIZES,active_heads)
                                  if active)
                    steps.append(FactorTransition(image.copy(),factors,0.,0.,
                                                  log_prob,action.ticks,ended,
                                                  active_heads=active_heads))
                    image=frames.append(view)
                    if ended:break
                path=save_rollout_batch(local,i,steps,dict(
                    schema=action_schema(args.ticks_per_decision),run_tag=args.run_tag,end_episode=i,episodes=[i],
                    stages=['random_normal_actions'],device='none',
                    image_shape=[6,84,84],action_interval=args.ticks_per_decision,
                    training_split=args.split))
                results['episodes']+=1
                results['decisions']+=len(steps)
                results['ticks']+=sum(s.duration for s in steps)
                results['natural_log_pickups']+=int(natural_log_count(bridge.packet)>0)
                manifest_file.write(json.dumps(dict(stage='episode',episode=i,
                    decisions=len(steps),file=str(path.relative_to(ROOT)),
                    bytes=path.stat().st_size))+'\n')
                manifest_file.flush()
                if i==1 or i%4==0 or i==args.episodes:
                    print('COLLECT:',dict(results),'last=',path.name,flush=True)
            return results
        outcomes=launch_world(f'MDL_VisualDynamics_{args.run_tag}',args,collect)
    print('COLLECTION COMPLETE',dict(outcomes),'data_dir=',local,flush=True)

if __name__=='__main__':main()
