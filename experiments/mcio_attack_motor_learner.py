"""Real Minecraft from-scratch attack-only actor-critic; held-out evaluation.

NARROW COMPONENT SKILL: controller stages a real oak tree and initial aim
outside the scored phase. The trained policy chooses ONLY holding/releasing
vanilla LEFT_BUTTON every six game ticks. It reads ONLY its OWN current
crosshair, visible client crack animation, elapsed ticks and last attack.
No teacher geometry/distance to target, pretrained weights, hidden chunk map,
server commands or fixture coordinate reaches the policy. Rewards during
training use own observed crack change and real mined-block statistics.
The scored/frozen check uses that SAME observation encoder and MCio game
extension. No claim of independent tree finding, pickup or Any% success.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.agent_interface import MCioAgentBridge, natural_log_count
from gmaincraft.joint_curriculum import sample_stationary_cases
from gmaincraft.mining_attack_policy import (MiningAttackLearner,AttackStep,
      SCHEMA,FEATURE_NAMES,action_for_attack,encode_attack_observation,step_reward)
from gmaincraft.instance_options import temporary_game_options
from joint_log_ppo import launch_world

MCIO=ROOT/'mcio-data'
INSTANCE=MCIO/'instances'/'DemoInstance'


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('train','eval'),required=True)
    parser.add_argument('--run-tag',required=True)
    parser.add_argument('--model-tag',default=None)
    parser.add_argument('--episodes',type=int,default=30)
    parser.add_argument('--eval-cases',type=int,default=12)
    parser.add_argument('--max-decisions',type=int,default=18)
    parser.add_argument('--train-seed',type=int,default=381231)
    parser.add_argument('--eval-seed',type=int,default=673231)
    parser.add_argument('--policy-seed',type=int,default=410233)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--visible',action='store_true')
    args=parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',args.run_tag):
        parser.error('Unsafe run tag')
    if args.model_tag and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',args.model_tag):
        parser.error('Unsafe model tag')
    if (args.episodes<1 or args.episodes>300 or args.eval_cases<1 or args.eval_cases>100
        or not 12<=args.max_decisions<=40 or len({args.train_seed,args.eval_seed,args.policy_seed})!=3):
        parser.error('Invalid independent episode/tick/seed bounds')
    if args.phase=='eval' and args.model_tag is None:
        parser.error('Frozen eval needs --model-tag')
    if args.phase=='train' and args.model_tag is not None:
        parser.error('Train always starts FROM SCRATCH; no --model-tag')
    # Existing shared launcher consumes these exact three properties.
    args.with_mining_feedback=True
    args.frame_width=320
    args.frame_height=180
    return args


def run_case(arena,case,policy,*,max_decisions,mode,random_rng=None):
    if mode not in ('learned','train_sample','untrained','always_attack','no_attack','random'):
        raise ValueError('Bad mode')
    started=time.perf_counter()
    arena.reset(case)  # controller fixture, strictly BEFORE scored actions
    time.sleep(.14)  # game SERVER ticks lag synchronized MCio client frames
    obs=arena.client.advance(12)
    bridge=MCioAgentBridge(arena.client.ctrl,obs)
    feedback=bridge.mining_feedback()
    status=bridge.player_status()
    initial_target=status.crosshair_block.block_id if status.crosshair_block else None
    baseline=bridge.stat_total('minecraft:mined','minecraft:oak_log')
    if natural_log_count(bridge.packet):
        raise AssertionError('Scored run started with nonempty wood inventory')
    prev_attack=False
    steps=[]
    stage_history=[]
    attack_decisions=0
    attack_streak=0
    longest_streak=0
    mined=False
    decisions=0
    for step in range(max_decisions):
        features=encode_attack_observation(status,feedback,
                        previous_attack=prev_attack,elapsed_ticks=step*6)
        if mode in ('learned','train_sample','untrained'):
            choice,prob=policy.act(features,greedy=(mode=='learned'))
        elif mode=='always_attack':
            choice,prob=1,1.
        elif mode=='no_attack':
            choice,prob=0,0.
        else:
            choice,prob=int(random_rng.random()<.5),.5
        old_feedback=feedback
        bridge.apply(action_for_attack(choice))
        feedback=bridge.mining_feedback()
        status=bridge.player_status()
        newly_mined=bridge.stat_total('minecraft:mined','minecraft:oak_log')>baseline
        reward=step_reward(old_feedback,feedback,mined_new_block=newly_mined)
        steps.append(AttackStep(features,choice,reward))
        stage_history.append(feedback.crack_stage)
        decisions+=1
        attack_decisions+=choice
        attack_streak=attack_streak+6 if choice else 0
        longest_streak=max(longest_streak,attack_streak)
        prev_attack=bool(choice)
        if newly_mined:
            mined=True
            break
    result=dict(mode=mode,mined=bool(mined),inventory_after_attack=natural_log_count(bridge.packet),
                first_crosshair=initial_target,decision_count=decisions,ticks=decisions*6,
                attack_decisions=attack_decisions,longest_attack_streak=longest_streak,
                max_crack_stage=max(stage_history,default=-1),
                reward_sum=round(sum(s.reward for s in steps),4),
                latest_attack_probability=round(float(prob),5),
                wall_s=round(time.perf_counter()-started,3),
                final_crosshair=status.crosshair_block.block_id if status.crosshair_block else None,
                stage_history=stage_history)
    print('ATTACK CASE',json.dumps(result,sort_keys=True),flush=True)
    return result,steps


def main(argv=None):
    args=parse_args(argv)
    tag=args.run_tag
    logs=ROOT/'logs'
    checkpoints=ROOT/'checkpoints'
    world_name=('MDL_AttackMotorTrain_' if args.phase=='train' else 'MDL_AttackMotorEval_')+tag
    destination=INSTANCE/'saves'/world_name
    output=(logs/f'mcio_attack_motor_train_{tag}.jsonl' if args.phase=='train'
            else logs/f'mcio_attack_motor_eval_{tag}.json')
    checkpoint=checkpoints/f'mcio_attack_motor_{tag}.pt'
    if destination.exists() or output.exists() or (args.phase=='train' and checkpoint.exists()):
        raise FileExistsError('Refusing to reuse existing world, log or model tag')
    case_seed=args.train_seed if args.phase=='train' else args.eval_seed
    config={'renderDistance':'4','maxFps':'260','enableVsync':'false'}
    device=args.device
    learner=MiningAttackLearner(device=device,seed=args.policy_seed)
    metadata=dict(schema=SCHEMA,feature_names=FEATURE_NAMES,
                  checkpoint_task='stationary_reachable_log_attack_only',
                  interval_ticks=6,init='random_weights',train_seed=args.train_seed,
                  eval_seed=args.eval_seed,policy_seed=args.policy_seed,
                  source='vanilla_MCio_opt_in_self_crack_stage',
                  training_reward='own_crack_stage_delta_plus_true_mined_event',
                  trainer_geometry_policy_fields=0,game_options=config)
    print('ATTACK MOTOR START',json.dumps(dict(tag=tag,phase=args.phase,
          options=config,device=str(learner.device),metadata=metadata)),flush=True)
    logs.mkdir(parents=True,exist_ok=True)
    checkpoints.mkdir(parents=True,exist_ok=True)
    with temporary_game_options(INSTANCE,config,run_tag=tag):
        if args.phase=='train':
            cases=sample_stationary_cases(case_seed,args.episodes)
            outcomes=[]
            with output.open('x',encoding='utf8') as stream:
                stream.write(json.dumps(dict(kind='config',phase=args.phase,
                         args=vars(args),metadata=metadata),default=str)+'\n')
                stream.flush()
                def train(arena):
                    for idx,case in enumerate(cases,1):
                        result,steps=run_case(arena,case,learner,
                                            max_decisions=args.max_decisions,
                                            mode='train_sample')
                        metrics=learner.update(steps)
                        outcomes.append(result)
                        stream.write(json.dumps(dict(kind='train',episode=idx,
                                      case=case.__dict__,result=result,update=metrics))+'\n')
                        stream.flush()
                        print('ATTACK UPDATE',idx,'mined',result['mined'],
                              'attack_decisions',result['attack_decisions'],
                              'metrics',metrics,flush=True)
                    return None
                launch_world(world_name,args,train)
                learner.save(checkpoint,metadata)
                digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                summary=dict(kind='summary',episodes=len(outcomes),
                        mined=sum(x['mined'] for x in outcomes),
                        first_half_mined=sum(x['mined'] for x in outcomes[:len(outcomes)//2]),
                        second_half_mined=sum(x['mined'] for x in outcomes[len(outcomes)//2:]),
                        updates=learner.updates,checkpoint_sha256=digest,
                        checkpoint_name=checkpoint.name)
                stream.write(json.dumps(summary)+'\n');stream.flush()
                print('ATTACK TRAIN SUMMARY',json.dumps(summary),flush=True)
                print('SAVED',checkpoint,output,flush=True)
        else:
            source=checkpoints/f'mcio_attack_motor_{args.model_tag}.pt'
            if not source.exists():raise FileNotFoundError('Trained checkpoint absent: '+str(source))
            actual=learner.load(source)
            if actual['schema']!=SCHEMA or actual['init']!='random_weights' or \
                    actual['train_seed']==args.eval_seed:
                raise ValueError('Invalid or non-independent attack motor checkpoint')
            # Independent random-weights control: no training data or weights
            # transferred from the trained neural policy.
            untrained=MiningAttackLearner(device=device,seed=args.policy_seed+99001)
            cases=sample_stationary_cases(case_seed,args.eval_cases)
            outcomes={}
            modes=('learned','untrained','always_attack','no_attack','random')
            rng=random.Random(args.eval_seed+1199)
            def evaluate(arena):
                for mode in modes:
                    subset=[]
                    for idx,case in enumerate(cases,1):
                        policy=(learner if mode=='learned' else untrained if mode=='untrained' else None)
                        record,_=run_case(arena,case,policy,
                               max_decisions=args.max_decisions,mode=mode,random_rng=rng)
                        record['case_index']=idx
                        subset.append(record)
                    outcomes[mode]=subset
                    print('ATTACK EVAL MODE',mode,'mined',sum(x['mined'] for x in subset),
                          'of',len(subset),flush=True)
            launch_world(world_name,args,evaluate)
            report=dict(tag=tag,model_tag=args.model_tag,task=metadata['checkpoint_task'],
                    model_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    policy_schema=SCHEMA,policy_inputs=FEATURE_NAMES,
                    same_encoder_train_eval=True,
                    eval_seed=args.eval_seed,train_seed=actual['train_seed'],
                    fixture_admin_only_before_episode=True,
                    learned_tree_navigation=False,learned_inventory_pickup=False,
                    gamescore_whole_run=False,
                    counts={mode:dict(mined=sum(x['mined'] for x in rows),n=len(rows))
                            for mode,rows in outcomes.items()},
                    results=outcomes)
            with output.open('x',encoding='utf8') as stream:
                json.dump(report,stream,indent=2);stream.write('\n')
            print('ATTACK EVAL SUMMARY',json.dumps(report['counts']),flush=True)
            print('SAVED',output,flush=True)

if __name__=='__main__':main()
