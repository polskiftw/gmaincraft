"""Paired REAL-Minecraft baseline checks for the V2 variable-log curriculum.

Uses SAME held-out `sample_near_cases(seed,n)` as factorized PPO. Every case
resets the disposable training world, kills dropped items and clears inventory.
Baseline policies are DIAGNOSTICS, not results of RL. `oracle_align` accesses
privileged target geometry to construct vanilla mouse motion, so it is an
upper-bound instrumentation check, NOT a fair pixel-only learner/baseline.
All scored interaction uses normal Minecraft input, never admin commands.
"""
import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.agent_interface import AgentAction, MCioAgentBridge,natural_log_count
from gmaincraft.joint_curriculum import sample_near_cases
from gmaincraft.vision_factorized import action_for_factors
from joint_log_ppo import launch_world

MODES=('attack_then_forward','forward_then_attack','random_factor','oracle_align')
DATA=ROOT/'logs'


def run_case(arena,case,mode,max_ticks,seed):
    ob=arena.reset(case)
    started=time.perf_counter()
    bridge=MCioAgentBridge(arena.client.ctrl,ob)
    rng=random.Random(seed)
    ticks=0;decisions=0
    before_distance,before_angle=arena.geometry(ob)
    def run(a):
        nonlocal ticks,decisions
        if ticks>=max_ticks or natural_log_count(bridge.packet)>0:
            return False
        allowed=max_ticks-ticks
        if a.ticks>allowed:a=AgentAction(a.keys,a.look_dx,a.look_dy,allowed)
        bridge.apply(a)
        ticks+=a.ticks;decisions+=1
        return True
    if mode=='oracle_align':
        # Teacher sees PRIVATE geometry, but only applies ordinary mouse input.
        # Purely an upper bound on fixture/input operability, NEVER a policy.
        tx,ty,tz=arena.target
        px,py,pz=ob.player_pos
        dx,dz=tx-px,tz-pz
        desired_yaw=math.degrees(math.atan2(-dx,dz))
        desired_pitch=-math.degrees(math.atan2(ty-(py+1.62),math.hypot(dx,dz)))
        dyaw=(desired_yaw-ob.player_yaw+180)%360-180
        dpitch=desired_pitch-ob.player_pitch
        run(AgentAction(look_dx=dyaw/.15,look_dy=dpitch/.15,ticks=3))
        run(AgentAction(ticks=3))
    if mode in ('attack_then_forward','oracle_align'):
        for i in range(20):run(AgentAction(keys=('LEFT_BUTTON',),ticks=6))
        while ticks<max_ticks and not natural_log_count(bridge.packet):
            run(AgentAction(keys=('W',),ticks=6))
    elif mode=='forward_then_attack':
        for i in range(2):run(AgentAction(keys=('W','LEFT_BUTTON'),ticks=6))
        for i in range(20):run(AgentAction(keys=('LEFT_BUTTON',),ticks=6))
        while ticks<max_ticks and not natural_log_count(bridge.packet):
            run(AgentAction(keys=('W',),ticks=6))
    elif mode=='random_factor':
        while ticks<max_ticks and not natural_log_count(bridge.packet):
            run(action_for_factors((rng.randrange(6),rng.randrange(5),rng.randrange(2))))
    else:raise ValueError(mode)
    d,angle=arena.geometry(bridge.packet)
    return dict(outcome='success' if natural_log_count(bridge.packet) else 'timeout',
                ticks=ticks,decisions=decisions,initial_distance=round(before_distance,2),
                initial_angle=round(before_angle,2),final_distance=round(d,2),
                final_angle=round(angle,2),inventory_logs=natural_log_count(bridge.packet),
                wall_seconds=round(time.perf_counter()-started,3))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--eval-seed',type=int,default=3737)
    ap.add_argument('--eval',type=int,default=12)
    ap.add_argument('--max-ticks',type=int,default=360)
    ap.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    ap.add_argument('--visible',action='store_true')
    ap.add_argument('--frame-width',type=int,default=320)
    ap.add_argument('--frame-height',type=int,default=180)
    args=ap.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',args.run_tag):ap.error('Invalid tag')
    if args.eval<1 or args.max_ticks<10:ap.error('Invalid counts')
    if not 96<=min(args.frame_width,args.frame_height) and max(args.frame_width,args.frame_height)<=1920:
        ap.error('Invalid viewport dimensions')
    output=DATA/f'factor_controls_{args.run_tag}.jsonl'
    save=ROOT/'mcio-data'/'instances'/'DemoInstance'/'saves'/f'MDL_FactorControls_{args.run_tag}'
    if output.exists() or save.exists():ap.error('Refusing overwrite')
    cases=sample_near_cases(args.eval_seed,args.eval)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8') as out:
        def evaluate(arena):
            from collections import Counter
            totals=Counter()
            # Interleave modes per case to ensure identical fixtures across
            # baselines rather than relying on different start populations.
            for i,case in enumerate(cases,1):
                for mode in args.modes:
                    result=run_case(arena,case,mode,args.max_ticks,
                                    seed=args.eval_seed+i*31)
                    totals[mode]+=int(result['outcome']=='success')
                    out.write(json.dumps(dict(mode=mode,case=i,**result))+'\n');out.flush()
                    print(f'CASE {i}/{len(cases)} MODE {mode}: {result}',flush=True)
            return totals
        totals=launch_world(f'MDL_FactorControls_{args.run_tag}',args,evaluate)
    print('BASELINE RESULTS:',dict(totals),'N each:',len(cases),flush=True)
    print('LOCAL:',output,flush=True)

if __name__=='__main__':main()
