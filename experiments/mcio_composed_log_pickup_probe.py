"""Frozen learned mining -> learned visual pickup, on actual mined item drops.

Controlled staged oak and pre-aimed player, ONLY before each scored episode.
Attack: frozen from-scratch game-memory crack-stage actor, or always/no attack.
Pickup: frozen older from-scratch RGB movement learner, or simple baselines.
Controller merely decides when the independently verified real Minecraft
'minecraft:mined' stat increased: this is an explicit evaluation handoff,
NOT learned end-to-end skill or world oracle. No summoned items or admin
commands during the scored phase. Score TRUE acquired inventory oak logs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.agent_interface import AgentAction,MCioAgentBridge,natural_log_count
from gmaincraft.instance_options import temporary_game_options
from gmaincraft.joint_curriculum import sample_stationary_cases
from gmaincraft.log_pickup import PickupPolicy,PICKUP_ACTIONS,pickup_features
from gmaincraft.mining_attack_policy import MiningAttackLearner,encode_attack_observation,action_for_attack,SCHEMA
from gmaincraft.reward_events import CumulativeMinedTracker
from joint_log_ppo import launch_world

MCIO=ROOT/'mcio-data'
INSTANCE=MCIO/'instances'/'DemoInstance'


def loaded_pickup(tag):
    path=ROOT/'checkpoints'/f'drop_pickup_policy_{tag}.npz'
    learner=PickupPolicy(seed=999)
    with np.load(path) as data:
        weights=data['weights'].copy()
    if weights.shape!=learner.weights.shape or not np.isfinite(weights).all():
        raise ValueError('Incompatible locally trained pickup model')
    learner.weights=weights
    return learner,hashlib.sha256(path.read_bytes()).hexdigest()


def play(arena,case,attack,*,mode,pickup,pickup_seed,max_attack_decisions,max_pickup_decisions,
         pickup_tracker):
    if mode not in ('learned_trained_pickup','always_trained_pickup','learned_forward',
                    'learned_random','learned_no_movement','no_attack_trained_pickup'):
        raise ValueError('Unsupported independently labeled baseline')
    arena.reset(case)
    time.sleep(.15)  # real server /setblock and client observations are not synced
    obs=arena.client.advance(12)
    bridge=MCioAgentBridge(arena.client.ctrl,obs)
    # Opt-in on-demand same-tick observation, after staged oak settles.
    bridge.apply(AgentAction(ticks=1),request_mining_feedback=True)
    if natural_log_count(bridge.packet):
        raise AssertionError('Fixture contamination: initial inventory already has wood')
    # IMPORTANT: MCio sends *cumulative* stats only on changed packets.
    # Each new per-episode bridge begins its cache at zero; treating its
    # first report as a delta would falsely count 1,2,3,... old mined blocks.
    # Persist the absolute world-session counter across independent resets.
    miner=arena.mined_events
    miner.begin_episode(bridge.stat_total('minecraft:mined','minecraft:oak_log'))
    pickup_tracker.begin_episode(bridge.stat_total('minecraft:picked_up','minecraft:oak_log'))
    real_mined_delta=0
    real_pickup_delta=0
    initial_oak=bridge.player_status().crosshair_block
    feedback=bridge.mining_feedback()
    status=bridge.player_status()
    prev_attack=False
    stage_trace=[]
    action_trace=[]
    mined=False
    start=time.perf_counter()
    attack_mode='learned' if mode.startswith('learned_') else 'always' if mode.startswith('always_') else 'never'
    for idx in range(max_attack_decisions):
        state=encode_attack_observation(status,feedback,
                              previous_attack=prev_attack,elapsed_ticks=idx*6)
        attack_choice=(attack.act(state,greedy=True)[0] if attack_mode=='learned'
                       else int(attack_mode=='always'))
        # Only ACTUAL vanilla press/release; no direct world mutation.
        bridge.apply(action_for_attack(attack_choice),request_mining_feedback=True)
        feedback=bridge.mining_feedback()
        status=bridge.player_status()
        stage_trace.append(feedback.crack_stage)
        action_trace.append(attack_choice)
        prev_attack=bool(attack_choice)
        real_mined_delta+=miner.observe(bridge.stat_total('minecraft:mined','minecraft:oak_log'))
        real_pickup_delta+=pickup_tracker.observe(bridge.stat_total('minecraft:picked_up','minecraft:oak_log'))
        if real_mined_delta:
            mined=True
            break
    wood_after_break=natural_log_count(bridge.packet)
    picked=wood_after_break>0
    pickup_trace=[]
    rng=random.Random(pickup_seed)
    # The evaluator announces phase switch only after a real Minecraft mined
    # event. It never exposes a drop position or synthetic item to the actor.
    if mined and not picked and mode!='learned_no_movement':
        for idx in range(max_pickup_decisions):
            if mode.endswith('trained_pickup'):
                frame=bridge.packet.get_frame()
                features=pickup_features(frame)
                movement,_=pickup.choose(features,rng,greedy=True)
            elif mode=='learned_forward':
                movement=0
            elif mode=='learned_random':
                movement=rng.randrange(len(PICKUP_ACTIONS))
            else:
                raise AssertionError('Invalid scored movement source')
            key=PICKUP_ACTIONS[movement]
            bridge.apply(AgentAction(keys=(key,),ticks=5))
            real_pickup_delta+=pickup_tracker.observe(bridge.stat_total('minecraft:picked_up','minecraft:oak_log'))
            pickup_trace.append(key)
            if natural_log_count(bridge.packet)>0:
                picked=True
                break
    result=dict(mode=mode,mined=bool(mined),inventory_success=bool(picked),
                first_target=initial_oak.block_id if initial_oak else None,
                attack_actions=len(action_trace),attack_hold_ticks=sum(action_trace)*6,
                attack_sequence=action_trace,crack_stages=stage_trace,
                movement_actions=len(pickup_trace),movement_trace=pickup_trace,
                inventory_after_mine=wood_after_break,
                inventory_end=natural_log_count(bridge.packet),
                mined_stat_delta=real_mined_delta,
                pickup_stat_delta=real_pickup_delta,
                final_position=[round(float(v),3) for v in bridge.packet.player_pos],
                scored_game_ticks=1+len(action_trace)*6+len(pickup_trace)*5,
                wall_s=round(time.perf_counter()-start,3),
                controller_commands_during_play=0)
    if picked and (not mined or result['inventory_end']<1):
        raise AssertionError('Invalid inventory success without a real mined oak log')
    if result['mined_stat_delta']!=int(mined) or result['pickup_stat_delta']>1:
        raise AssertionError('Stale or duplicated cumulative mining/pickup stats')
    print('COMPOSED CASE',json.dumps(result,sort_keys=True),flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-tag',required=True)
    parser.add_argument('--attack-model',default='motorlearn_nov03a')
    parser.add_argument('--pickup-model',default='p02')
    parser.add_argument('--cases',type=int,default=12)
    parser.add_argument('--eval-seed',type=int,default=784231)
    parser.add_argument('--max-attack-decisions',type=int,default=18)
    parser.add_argument('--max-pickup-decisions',type=int,default=22)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='cuda')
    parser.add_argument('--visible',action='store_true')
    args=parser.parse_args(argv)
    for name in (args.run_tag,args.attack_model,args.pickup_model):
        if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,31}',name):
            parser.error('Unsafe tag/model name')
    if not (1<=args.cases<=80 and 1<=args.max_attack_decisions<=30 and
            1<=args.max_pickup_decisions<=60 and args.eval_seed!=381231):
        parser.error('Invalid episode duration or seed')
    args.with_mining_feedback=True
    args.with_mining_on_demand=True
    args.frame_width=320
    args.frame_height=180
    world_name='MDL_ComposedMinePickup_'+args.run_tag
    output=ROOT/'logs'/f'mcio_composed_mined_pickup_{args.run_tag}.json'
    if output.exists() or (INSTANCE/'saves'/world_name).exists():
        parser.error('Refusing overwrite of an existing world or report')
    checkpoint=ROOT/'checkpoints'/f'mcio_attack_motor_{args.attack_model}.pt'
    model=MiningAttackLearner(device=args.device,seed=410233)
    metadata=model.load(checkpoint)
    if metadata['schema']!=SCHEMA or metadata['train_seed']==args.eval_seed:
        raise ValueError('Learned checkpoint lacks independent eval schema/seeds')
    checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    pickup,pickup_sha=loaded_pickup(args.pickup_model)
    original_pickup_weights=pickup.weights.copy()
    cases=sample_stationary_cases(args.eval_seed,args.cases)
    modes=('learned_trained_pickup','always_trained_pickup',
           'learned_forward','learned_random','learned_no_movement',
           'no_attack_trained_pickup')
    settings={'renderDistance':'4','maxFps':'260','enableVsync':'false'}
    print('COMPOSED START',json.dumps(dict(tag=args.run_tag,attack_sha256=checkpoint_sha,
         pickup_sha256=pickup_sha,cases=args.cases,seed=args.eval_seed,modes=modes,
         game_settings=settings)),flush=True)
    results={}
    def evaluate(arena):
        pickup_tracker=CumulativeMinedTracker()
        for mode in modes:
            samples=[]
            for idx,case in enumerate(cases,1):
                sample=play(arena,case,model,mode=mode,pickup=pickup,
                            pickup_seed=args.eval_seed+idx*731,
                            max_attack_decisions=args.max_attack_decisions,
                            max_pickup_decisions=args.max_pickup_decisions,
                            pickup_tracker=pickup_tracker)
                sample['case_index']=idx
                samples.append(sample)
            results[mode]=samples
            print('COMPOSED MODE',mode,'mined',sum(x['mined'] for x in samples),
                  'picked',sum(x['inventory_success'] for x in samples),
                  'cases',len(samples),flush=True)
        return None
    with temporary_game_options(INSTANCE,settings,run_tag=args.run_tag):
        launch_world(world_name,args,evaluate)
    if not np.array_equal(pickup.weights,original_pickup_weights):
        raise AssertionError('Frozen pickup model weights were unexpectedly changed')
    counts={name:dict(n=len(rows),mined=sum(x['mined'] for x in rows),
             actual_inventory_pickup=sum(x['inventory_success'] for x in rows),
             pickups_needing_movement=sum(x['inventory_success'] and x['inventory_after_mine']==0
                                         for x in rows)) for name,rows in results.items()}
    report=dict(tag=args.run_tag,task='preaimed_mining_then_mined_drop_inventory_pickup',
                checkpoint_sha256=checkpoint_sha,pickup_checkpoint_sha256=pickup_sha,
                same_mining_schema=SCHEMA,pickup_model_tag=args.pickup_model,
                independent_eval_seed=args.eval_seed,trained_attack_seed=metadata['train_seed'],
                scored_gameplay_has_admin_commands=False,
                item_was_summoned_during_scored_play=False,
                fixture_uses_controller_setup_before_play=True,
                phase_handoff_is_scripted_on_real_mined_stat=True,
                learned_end_to_end=False,general_natural_world_success=False,
                game_settings=settings,counts=counts,results=results)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8') as stream:
        json.dump(report,stream,indent=2)
        stream.write('\n')
    print('COMPOSED SUMMARY',json.dumps(counts,sort_keys=True),flush=True)
    print('SAVED',output,flush=True)

if __name__=='__main__':main()
