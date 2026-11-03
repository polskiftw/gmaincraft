"""V2 joint camera+movement+attack PPO on REAL Minecraft, 6-tick intervals.

Unlike V1's fixed 80-tick mine/no-motion macro, this model picks movement,
look and attack independently every 6 game ticks, holding LEFT_BUTTON across
consecutive attack decisions. Learn from random RGB CNN weights, using full
trajectory PPO with teacher PRIVILEGED scalar shaping only during disposable
flat-world training. Controller may reset variable placed-log worlds between
episodes, not during scored gameplay. Success is observed vanilla inventory.

Frozen evaluation runs in separately copied world and never updates weights.
Natural-world frozen transfer is a distinct real-seed benchmark mode.
"""
import argparse
import json
from collections import Counter
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.agent_interface import MCioAgentBridge,natural_log_count
from gmaincraft.joint_curriculum import (sample_cases,sample_near_cases,
                                         sample_stationary_cases,interleave_phases)
from gmaincraft.vision_ppo import PixelHistory
from gmaincraft.rollout_store import save_rollout_batch
from gmaincraft.inverse_dynamics import load_encoder_checkpoint
from gmaincraft.reward_events import mined_event_reward
from gmaincraft.memory_factorized import (MemoryFactorizedPPOTrainer,
    MemoryFactorTransition, encode_player_knowledge)
from gmaincraft.vision_factorized import (FactorizedPPOTrainer,FactorTransition,
                                           ACTION_TICKS,SCHEMA,action_schema,action_for_factors)
from joint_log_ppo import launch_world

MCIO=ROOT/'mcio-data'


def reward_for_step(before,after,factors,ticks,*,success=False,dead=False,
                    stable_ticks=0,decision_ticks=ACTION_TICKS):
    """TRAINER-side scalar shaping from privileged position and target bearing.

    Logs are scored ONLY from real inventory; geometry is not a policy input.
    Short-interval attack reward helps bootstrap hold persistence when truly
    aiming at an in-reach trunk, rather than waiting for a rare full break.
    """
    bd,be=before
    nd,ne=after
    # Positive when approaching a log; penalize retreating or standing still.
    dist_reward=.38*float(np.clip(bd-nd,-.9,.9))
    look_reward=.020*float(np.clip(be-ne,-30.,30.))
    press=bool(factors[2])
    # Distance/angle are oracle measurements ONLY for a scalar train reward;
    # this encourages holding attack while aligned, not bogus spam at spawn.
    if press:
        if nd<=4.1 and ne<=9:
            attack_reward=.065*(ticks/decision_ticks)
        elif nd<=4.1 and ne<=18:
            attack_reward=.015*(ticks/decision_ticks)
        else:
            attack_reward=-.020*(ticks/decision_ticks)
    else:
        attack_reward=0.
    # Private teacher proxy for maintaining aim *through multiple actions*.
    # This is not an actual Minecraft block-crack metric or a success label.
    # Reward increases modestly with continuous well-aimed attack, rather
    # than making a one-frame flyby as valuable as sustained mining.
    stable_bonus=(min(.20,.002*stable_ticks) if press and stable_ticks>0 else 0.)
    return (dist_reward+look_reward+attack_reward+stable_bonus
            -.010*(ticks/decision_ticks)
            +(20.0 if success else 0.0)
            -(8.0 if dead else 0.0))


def run_episode(arena,agent,case,*,max_ticks,training,active_heads=(True,True,True),
                decision_ticks=ACTION_TICKS,mine_event_bonus=0.):
    ob=arena.reset(case)
    bridge=MCioAgentBridge(arena.client.ctrl,ob)
    memory_mode=isinstance(agent,MemoryFactorizedPPOTrainer)
    history=PixelHistory()
    initial=bridge.multimodal_view() if memory_mode else bridge.view()
    image=history.reset(initial)
    features=encode_player_knowledge(initial.player) if memory_mode else None
    event_tracker=arena.mined_events
    event_tracker.begin_episode(bridge.stat_total('minecraft:mined','minecraft:oak_log'))
    total_event_mined=0
    before=arena.geometry(ob)
    initial_distance,initial_angle=before
    best_angle=initial_angle
    aligned_ticks=0
    aligned_attack_ticks=0
    stable_ticks=0
    longest_stable_ticks=0
    ticks=0;decisions=0;total=0.;streak=0;max_streak=0
    counter=Counter()
    trajectory=[]
    started=time.perf_counter()
    time_inference=time_game=time_resize=0.
    success=False;dead=False
    while ticks<max_ticks:
        clock=time.perf_counter()
        if memory_mode:
            factors,logp,value=agent.act(image,features,greedy=not training,
                                         active_heads=active_heads)
        else:
            factors,logp,value=agent.act(image,greedy=not training,active_heads=active_heads)
        time_inference+=time.perf_counter()-clock
        action=action_for_factors(factors,max_ticks-ticks,decision_ticks=decision_ticks)
        clock=time.perf_counter()
        next_view=bridge.apply(action)
        if memory_mode:
            # The bridge's same read-only opt-in view is used during real-game
            # training and frozen scored evaluation (including natural seeds).
            next_view=bridge.multimodal_view()
        time_game+=time.perf_counter()-clock
        ticks+=action.ticks;decisions+=1
        after=arena.geometry(bridge.packet)
        success=natural_log_count(bridge.packet)>0
        dead=bridge.packet.health<=0
        best_angle=min(best_angle,after[1])
        if after[0]<=4.1 and after[1]<=9:
            aligned_ticks+=action.ticks
            if factors[2]: aligned_attack_ticks+=action.ticks
        if (factors[2] and before[0]<=4.1 and after[0]<=4.1 and
                before[1]<=8 and after[1]<=8):
            stable_ticks+=action.ticks
            longest_stable_ticks=max(longest_stable_ticks,stable_ticks)
        else:
            stable_ticks=0
        reward=reward_for_step(before,after,factors,action.ticks,
                               success=success,dead=dead,stable_ticks=stable_ticks,
                               decision_ticks=decision_ticks)
        mined_total=bridge.stat_total('minecraft:mined','minecraft:oak_log')
        new_blocks=event_tracker.observe(mined_total)
        if new_blocks:
            total_event_mined+=new_blocks
            if training and mine_event_bonus:
                reward+=mined_event_reward(0,new_blocks,reward_per_block=mine_event_bonus)
        done=success or dead or ticks>=max_ticks
        if training:
            if memory_mode:
                trajectory.append(MemoryFactorTransition(image.copy(),features.copy(),
                    factors,float(reward),value,logp,action.ticks,done,active_heads))
            else:
                trajectory.append(FactorTransition(image.copy(),factors,float(reward),
                                                   value,logp,action.ticks,done,active_heads))
        total+=reward
        counter['attack_ticks']+=action.ticks*int(factors[2]==1)
        counter['move_ticks']+=action.ticks*int(factors[0]!=0)
        counter['look_decisions']+=int(factors[1]!=0)
        counter['attack_decisions']+=int(factors[2]!=0)
        counter[f'move_{factors[0]}']+=1
        counter[f'look_{factors[1]}']+=1
        if factors[2]:
            streak+=action.ticks
            max_streak=max(max_streak,streak)
        else:streak=0
        before=after
        clock=time.perf_counter()
        image=history.append(next_view)
        if memory_mode:
            features=encode_player_knowledge(next_view.player)
        time_resize+=time.perf_counter()-clock
        if done:break
    outcome='success' if success else 'dead' if dead else 'timeout'
    result=dict(outcome=outcome,ticks=ticks,decisions=decisions,
                reward=round(total,4),mined_oak_events=total_event_mined,
                initial_distance=round(initial_distance,3),
                initial_angle=round(initial_angle,3),
                end_distance=round(before[0],3),end_angle=round(before[1],3),
                best_angle=round(best_angle,3),aligned_ticks=aligned_ticks,
                aligned_attack_ticks=aligned_attack_ticks,
                longest_stable_attack_ticks=longest_stable_ticks,
                attack_ticks=counter['attack_ticks'],
                move_ticks=counter['move_ticks'],
                look_decisions=counter['look_decisions'],
                max_attack_streak_ticks=max_streak,
                action_hist={k:counter[k] for k in sorted(counter) if k.startswith(('move_','look_'))},
                wall_seconds=round(time.perf_counter()-started,3),
                policy_inference_wall_s=round(time_inference,4),
                game_io_wall_s=round(time_game,4),
                rgb_processing_wall_s=round(time_resize,4))
    if memory_mode:
        result['observation_schema']=agent.observation_schema
    return result,trajectory


def parse_args(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--episodes',type=int,default=120)
    ap.add_argument('--eval',type=int,default=12,help='cases PER near/broad split')
    ap.add_argument('--max-ticks',type=int,default=360)
    ap.add_argument('--update-every',type=int,default=10)
    ap.add_argument('--mine-event-bonus',type=float,default=0.0,
                    help='extra reward for actual oak blocks mined (training only); 0 disables')
    ap.add_argument('--decision-ticks',type=int,choices=(6,12),default=6,
                    help='new policy/rollout schema: 6 old V2 or 12 experimental V3')
    ap.add_argument('--curriculum',choices=['near','broad','mix','staged'],default='staged')
    ap.add_argument('--stationary-fraction',type=float,default=.25)
    ap.add_argument('--schedule',choices=['blocked','interleaved'],default='interleaved')
    ap.add_argument('--aim-fraction',type=float,default=.40,
                    help='fraction of post-stationary cases with movement locked, look+attack enabled')
    ap.add_argument('--near-fraction',type=float,default=.7)
    ap.add_argument('--train-seed',type=int,default=27031)
    ap.add_argument('--eval-seed',type=int,default=74013)
    ap.add_argument('--policy-seed',type=int,default=20141)
    ap.add_argument('--init-model',default=None)
    ap.add_argument('--vision-init-tag',default=None,
                    help='own self-collected RGB inverse-dynamics checkpoint, not external weights')
    ap.add_argument('--device',choices=['cpu','cuda','auto'],default='cuda')
    ap.add_argument('--frame-width',type=int,default=320)
    ap.add_argument('--frame-height',type=int,default=180)
    ap.add_argument('--visible',action='store_true')
    ap.add_argument('--skip-eval',action='store_true')
    ap.add_argument('--save-rollouts',action='store_true',
                    help='save local ignored RGB/action/reward trajectory NPZ per PPO update')
    args=ap.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',args.run_tag):ap.error('Invalid run tag')
    if min(args.episodes,args.eval,args.max_ticks,args.update_every)<1:ap.error('Counts must be >0')
    if not 0<=args.near_fraction<=1:ap.error('near-fraction must be 0..1')
    if not 0<=args.stationary_fraction<=1:ap.error('stationary-fraction must be 0..1')
    if not 0<=args.aim_fraction<=1:ap.error('aim-fraction must be 0..1')
    if min(args.frame_width,args.frame_height)<96 or max(args.frame_width,args.frame_height)>1920:
        ap.error('Frame dimensions must be 96..1920')
    if not 0<=args.mine_event_bonus<=20:ap.error('Event bonus must be 0..20')
    if args.train_seed==args.eval_seed:ap.error('Train and eval seeds must differ')
    if args.init_model and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',args.init_model):
        ap.error('Invalid init-model')
    if args.init_model==args.run_tag:ap.error('Warm starts need new output tags')
    if args.init_model and args.vision_init_tag:
        ap.error('Cannot warm start RL weights AND separately initialize visual weights')
    if args.vision_init_tag and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',args.vision_init_tag):
        ap.error('Invalid vision init tag')
    return args


def training_plan(args):
    if args.curriculum=='staged':
        stationary=int(round(args.episodes*args.stationary_fraction))
        remain=args.episodes-stationary
        aim=int(round(remain*args.aim_fraction))
        remaining_full=remain-aim
        near=int(round(remaining_full*args.near_fraction))
        plan=[]
        if stationary:
            plan.extend([('stationary',case,(False,False,True))
                         for case in sample_stationary_cases(args.train_seed,stationary)])
        if aim:
            plan.extend([('aim',case,(False,True,True))
                         for case in sample_near_cases(args.train_seed+1,aim)])
        if near:
            plan.extend([('near',case,(True,True,True))
                         for case in sample_near_cases(args.train_seed+2,near)])
        if remaining_full-near:
            plan.extend([('broad',case,(True,True,True))
                         for case in sample_cases(args.train_seed+3,remaining_full-near)])
        return interleave_phases(plan) if args.schedule=='interleaved' else plan
    if args.curriculum=='near':return [('near',case,(True,True,True)) for case in sample_near_cases(args.train_seed,args.episodes)]
    if args.curriculum=='broad':return [('broad',case,(True,True,True)) for case in sample_cases(args.train_seed,args.episodes)]
    count=int(round(args.episodes*args.near_fraction))
    plan=[]
    if count:plan.extend([('near',case,(True,True,True)) for case in sample_near_cases(args.train_seed,count)])
    if args.episodes-count:
        plan.extend([('broad',case,(True,True,True)) for case in sample_cases(args.train_seed+1,args.episodes-count)])
    return plan


def main(argv=None):
    args=parse_args(argv)
    train_name='MDL_FactorTrain_'+args.run_tag
    eval_name='MDL_FactorEval_'+args.run_tag
    log=ROOT/'logs'/f'factorized_ppo_{args.run_tag}.jsonl'
    cp=ROOT/'checkpoints'/f'joint_factorized_{args.run_tag}.pt'
    saves=MCIO/'instances'/'DemoInstance'/'saves'
    if log.exists() or cp.exists() or (saves/train_name).exists() or (saves/eval_name).exists():
        raise FileExistsError('Run tag collision: refusing to overwrite checkpoints/results/worlds')
    model=FactorizedPPOTrainer(device=args.device,seed=args.policy_seed)
    schema=action_schema(args.decision_ticks)
    ancestor=None
    if args.vision_init_tag:
        path=ROOT/'checkpoints'/f'visual_inverse_{args.vision_init_tag}.pt'
        visual,metadata=load_encoder_checkpoint(path)
        if metadata.get('pixel_encoding','raw')!='raw':
            raise ValueError('A delta-encoded encoder cannot initialize a raw-RGB policy')
        model.net.visual.load_state_dict(visual)
        print('INIT VISUAL ENCODER FROM OWN RANDOM-PLAY SCREENSHOTS',path,flush=True)
    if args.init_model:
        path=ROOT/'checkpoints'/f'joint_factorized_{args.init_model}.pt'
        ancestor=model.load_weights(path,restore_optimizer=True)
        if ancestor.get('schema')!=schema:
            raise ValueError('Incompatible saved factorized action policy')
        if args.train_seed==ancestor.get('train_seed') or args.policy_seed==ancestor.get('policy_seed'):
            raise ValueError('Warm start must use new training and policy seeds')
    train_cases=training_plan(args)
    evals=[('stationary_restricted',sample_stationary_cases(args.eval_seed+2,args.eval),
             (False,False,True)),
           ('aim_restricted',sample_near_cases(args.eval_seed+3,args.eval),
             (False,True,True)),
           ('near',sample_near_cases(args.eval_seed,args.eval),(True,True,True)),
           ('broad',sample_cases(args.eval_seed+1,args.eval),(True,True,True))]
    print(f'V2 FACTORIZED PPO device={model.device} params={model.parameter_count} '
          f'episodes={len(train_cases)} max_ticks={args.max_ticks} '
          f'update_every={args.update_every} actions=move6 x look5 x attack2, '
          f'{args.decision_ticks} ticks/decision; curriculum={args.curriculum}; schedule={args.schedule}',flush=True)
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x',encoding='utf8') as file:
        config=dict(stage='config',tag=args.run_tag,device=str(model.device),
                    schema=schema,decision_ticks=args.decision_ticks,
                    policy_seed=args.policy_seed,train_seed=args.train_seed,
                    eval_seed=args.eval_seed,init_model=args.init_model,
                    vision_init_tag=args.vision_init_tag,
                    curriculum=args.curriculum,near_fraction=args.near_fraction,
                    stationary_fraction=args.stationary_fraction,
                    aim_fraction=args.aim_fraction,schedule=args.schedule,
                    reward_schema='v2_stable_attack_teacher_proxy_plus_real_mining',
                    mine_event_bonus=args.mine_event_bonus,
                    max_ticks=args.max_ticks,episodes=args.episodes,eval=args.eval,
                    frame_width=args.frame_width,frame_height=args.frame_height,
                    save_rollouts=args.save_rollouts)
        file.write(json.dumps(config)+'\n');file.flush()
        def train(arena):
            traj=[];batch_episodes=[];batch_phases=[];total=Counter()
            for i,(phase,case,mask) in enumerate(train_cases,1):
                result,steps=run_episode(arena,model,case,max_ticks=args.max_ticks,
                                         training=True,active_heads=mask,
                                         decision_ticks=args.decision_ticks,
                                         mine_event_bonus=args.mine_event_bonus)
                total[phase]+=1;total[phase+'_success']+=int(result['outcome']=='success')
                file.write(json.dumps(dict(stage='train',episode=i,phase=phase,active_heads=mask,case=case.__dict__,**result))+'\n')
                file.flush();traj.extend(steps)
                batch_episodes.append(i);batch_phases.append(phase)
                if i%args.update_every==0 or i==len(train_cases):
                    if args.save_rollouts:
                        filename=save_rollout_batch(
                            ROOT/'runs'/f'joint_factorized_{args.run_tag}',i,traj,
                            dict(schema=schema,run_tag=args.run_tag,end_episode=i,
                                 episodes=batch_episodes,stages=batch_phases,
                                 device=str(model.device),training_split='train',
                                 image_shape=[6,84,84],action_interval=args.decision_ticks))
                        file.write(json.dumps(dict(stage='rollout',episode=i,
                                                   file=str(filename.relative_to(ROOT)),
                                                   transitions=len(traj),
                                                   bytes=filename.stat().st_size))+'\n')
                        file.flush()
                    stats=model.update(traj);traj=[]
                    batch_episodes=[];batch_phases=[]
                    model.save(cp,dict(schema=schema,decision_ticks=args.decision_ticks,
                                       policy_seed=args.policy_seed,train_seed=args.train_seed,
                                       eval_seed=args.eval_seed,episodes_completed=i,
                                       initialized_from=args.init_model or 'scratch',
                                       vision_init_tag=args.vision_init_tag,
                                       curriculum=args.curriculum,max_ticks=args.max_ticks,
                                       mine_event_bonus=args.mine_event_bonus))
                    file.write(json.dumps(dict(stage='update',episode=i,**stats))+'\n');file.flush()
                    print(f'UPDATE {i}/{args.episodes} stationary={total["stationary_success"]}/{total["stationary"]} '
                          f'aim={total["aim_success"]}/{total["aim"]} '
                          f'near={total["near_success"]}/{total["near"]} '
                          f'broad={total["broad_success"]}/{total["broad"]} '
                          f'optim={stats} latest={result}',flush=True)
            return total
        started=time.perf_counter()
        total=launch_world(train_name,args,train)
        print('TRAIN DONE:',dict(total),'elapsed_wall=',round(time.perf_counter()-started,2),flush=True)
        if not args.skip_eval:
            def evaluate(arena):
                total=Counter()
                for phase,cases,mask in evals:
                    for i,case in enumerate(cases,1):
                        result,_=run_episode(arena,model,case,max_ticks=args.max_ticks,
                                             training=False,active_heads=mask,
                                             decision_ticks=args.decision_ticks)
                        total[phase]+=1;total[phase+'_success']+=int(result['outcome']=='success')
                        file.write(json.dumps(dict(stage='eval',phase=phase,active_heads=mask,case_index=i,
                                                   case=case.__dict__,**result))+'\n');file.flush()
                        print(f'EVAL {phase} {i}/{len(cases)} {result}',flush=True)
                return total
            evaluated=launch_world(eval_name,args,evaluate)
            print('FROZEN EVAL:',dict(evaluated),flush=True)
    print('LOCAL CHECKPOINT:',cp,flush=True)
    print('LOCAL JSONL:',log,flush=True)

if __name__=='__main__':main()
