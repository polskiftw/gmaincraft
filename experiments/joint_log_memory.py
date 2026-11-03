"""V4 from-scratch Minecraft PPO: own memory-state + RGB policy inputs.

The SAME bounded MCio multimodal observation interface is used for controlled
training and frozen evaluation. Controller geometry is TRAINING REWARD ONLY.
The policy must take vanilla actions to mine and actually collect inventory.
Does not claim a learned tree finder, crafting or Any% speedrun capability.

Run with a unique --run-tag; local worlds/logs/checkpoint are never reused.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.instance_options import temporary_game_options
from gmaincraft.memory_factorized import (MemoryFactorizedPPOTrainer,SCHEMA,
    FEATURE_NAMES,FEATURE_SIZE)
from gmaincraft.joint_curriculum import sample_near_cases,sample_cases,sample_stationary_cases
from joint_log_factorized import run_episode,training_plan
from joint_log_ppo import launch_world

MCIO=ROOT/'mcio-data'


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-tag',required=True)
    parser.add_argument('--episodes',type=int,default=12)
    parser.add_argument('--eval',type=int,default=3,help='held-out cases per difficulty')
    parser.add_argument('--max-ticks',type=int,default=240)
    parser.add_argument('--update-every',type=int,default=6)
    parser.add_argument('--mine-event-bonus',type=float,default=3.0)
    parser.add_argument('--train-seed',type=int,default=210031)
    parser.add_argument('--eval-seed',type=int,default=310071)
    parser.add_argument('--policy-seed',type=int,default=110013)
    parser.add_argument('--device',choices=('cuda','cpu','auto'),default='cuda')
    parser.add_argument('--frame-width',type=int,default=320)
    parser.add_argument('--frame-height',type=int,default=180)
    parser.add_argument('--visible',action='store_true')
    parser.add_argument('--render-distance',type=int,default=4,
                        help='dedicated MCio instance only; 4 chunks is measured fast profile')
    parser.add_argument('--game-fps',type=int,default=260)
    parser.add_argument('--game-vsync',choices=('on','off','keep'),default='off')
    parser.add_argument('--skip-eval',action='store_true')
    parser.add_argument('--curriculum',choices=('staged','near','broad','mix'),default='staged')
    parser.add_argument('--stationary-fraction',type=float,default=.25)
    parser.add_argument('--aim-fraction',type=float,default=.40)
    parser.add_argument('--near-fraction',type=float,default=.70)
    parser.add_argument('--schedule',choices=('blocked','interleaved'),default='interleaved')
    parser.add_argument('--warn-slow-tick-ms',type=float,default=10,
                        help='warn when measured MCio I/O per game tick exceeds this')
    args=parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',args.run_tag):
        parser.error('Invalid run-tag')
    if not 2<=args.render_distance<=32 or not 10<=args.game_fps<=260:
        parser.error('Invalid temporary Minecraft render distance/FPS')
    if min(args.episodes,args.eval,args.max_ticks,args.update_every)<1:
        parser.error('Counts must be >0')
    if not 96 <= min(args.frame_width,args.frame_height) and max(args.frame_width,args.frame_height)<=1920:
        parser.error('Expected frame dimensions 96..1920')
    if len({args.train_seed,args.eval_seed,args.policy_seed})!=3:
        parser.error('Train/eval/policy seeds must be distinct')
    if not all(0<=x<=1 for x in (args.stationary_fraction,args.aim_fraction,args.near_fraction)):
        parser.error('Curriculum fractions must lie between 0 and 1')
    if not 0<=args.mine_event_bonus<=20:
        parser.error('Invalid mine event bonus')
    if args.warn_slow_tick_ms<=0:
        parser.error('Invalid slow game tick threshold')
    return args


def check_speed(record,args):
    tick_ms=record['game_io_wall_s']*1000/record['ticks']
    if tick_ms > args.warn_slow_tick_ms:
        print(f'WARNING: MCio game I/O {tick_ms:.2f} ms/tick exceeds '
              f'{args.warn_slow_tick_ms:g}; compare no-Torch baseline with same host state',flush=True)
    return round(tick_ms,4)


def main(argv=None):
    args=parse_args(argv)
    train_world='MDL_MemoryTrain_'+args.run_tag
    eval_world='MDL_MemoryEval_'+args.run_tag
    saves=MCIO/'instances'/'DemoInstance'/'saves'
    log=ROOT/'logs'/f'memory_ppo_{args.run_tag}.jsonl'
    checkpoint=ROOT/'checkpoints'/f'joint_memory_{args.run_tag}.pt'
    if any(p.exists() for p in (saves/train_world,saves/eval_world,log,checkpoint)):
        raise FileExistsError('Would overwrite an existing local run: choose a new tag')
    overrides={'renderDistance':str(args.render_distance),'maxFps':str(args.game_fps)}
    if args.game_vsync!='keep': overrides['enableVsync']='true' if args.game_vsync=='on' else 'false'
    model=MemoryFactorizedPPOTrainer(device=args.device,seed=args.policy_seed)
    cases=training_plan(args)
    eval_sets=(('stationary_restricted',sample_stationary_cases(args.eval_seed+2,args.eval),(False,False,True)),
               ('near_unrestricted',sample_near_cases(args.eval_seed,args.eval),(True,True,True)),
               ('broad_unrestricted',sample_cases(args.eval_seed+1,args.eval),(True,True,True)))
    metadata=dict(schema=SCHEMA,feature_names=FEATURE_NAMES,feature_size=FEATURE_SIZE,
                  policy_seed=args.policy_seed,train_seed=args.train_seed,
                  eval_seed=args.eval_seed,curriculum=args.curriculum,
                  decision_ticks=6,reward_schema='v4_player_memory_rgb_teacher_scalar',
                  mine_event_bonus=args.mine_event_bonus,frame_size=[args.frame_width,args.frame_height],
                  observations='MCioAgentBridge.multimodal_view',
                  state_schema='player_knowledge_v1',initialized_from='scratch',
                  game_options=overrides)
    print('V4 MEMORY+RGB PPO',json.dumps(dict(args=vars(args),metadata=metadata),default=list),flush=True)
    log.parent.mkdir(parents=True,exist_ok=True)
    with temporary_game_options(MCIO/'instances'/'DemoInstance',overrides,run_tag=args.run_tag), \
            log.open('x',encoding='utf-8') as file:
        file.write(json.dumps(dict(stage='config',args=vars(args),model=metadata))+'\n');file.flush()
        def train(arena):
            transitions=[]
            totals=Counter()
            for episode,(phase,case,mask) in enumerate(cases,1):
                record,steps=run_episode(arena,model,case,max_ticks=args.max_ticks,
                                         training=True,active_heads=mask,
                                         decision_ticks=6,mine_event_bonus=args.mine_event_bonus)
                totals[phase]+=1
                totals[phase+'_success']+=int(record['outcome']=='success')
                ms=check_speed(record,args)
                file.write(json.dumps(dict(stage='train',episode=episode,phase=phase,
                                           active_heads=mask,case=case.__dict__,
                                           io_ms_per_tick=ms,**record))+'\n');file.flush()
                transitions.extend(steps)
                if episode%args.update_every==0 or episode==len(cases):
                    metrics=model.update(transitions)
                    transitions=[]
                    model.save(checkpoint,dict(metadata,episodes_completed=episode,**metrics))
                    file.write(json.dumps(dict(stage='update',episode=episode,**metrics))+'\n');file.flush()
                    print(f'UPDATE {episode}/{len(cases)} io_ms_per_tick={ms} '
                          f'outcome={record["outcome"]} metrics={metrics}',flush=True)
            return dict(totals)
        started=time.perf_counter()
        results=launch_world(train_world,args,train)
        print('V4 TRAIN',results,'wall_s',round(time.perf_counter()-started,2),flush=True)
        if not args.skip_eval:
            def evaluate(arena):
                totals=Counter()
                for phase,scenarios,mask in eval_sets:
                    for i,case in enumerate(scenarios,1):
                        record,_=run_episode(arena,model,case,max_ticks=args.max_ticks,
                                             training=False,active_heads=mask,
                                             decision_ticks=6,mine_event_bonus=0.)
                        ms=check_speed(record,args)
                        totals[phase]+=1
                        totals[phase+'_success']+=int(record['outcome']=='success')
                        file.write(json.dumps(dict(stage='eval',phase=phase,case_index=i,
                                                   active_heads=mask,case=case.__dict__,
                                                   io_ms_per_tick=ms,**record))+'\n');file.flush()
                        print(f'FROZEN {phase} case={i} result={record["outcome"]} '
                              f'mined={record["mined_oak_events"]} io_ms/tick={ms}',flush=True)
                return dict(totals)
            eval_result=launch_world(eval_world,args,evaluate)
            print('V4 FROZEN EVAL',eval_result,flush=True)
    print('V4 SAVED',checkpoint,'SHA256',hashlib.sha256(checkpoint.read_bytes()).hexdigest(),flush=True)
    print('V4 LOG',log,flush=True)

if __name__=='__main__':main()
