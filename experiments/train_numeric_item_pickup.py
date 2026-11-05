"""From-scratch numeric dropped-item pickup skill in a disposable Minecraft lab.

The skill receives ONLY own player state + current camera-visible ItemEntity
observations + timestamped memory. It chooses normal vanilla key/mouse
actions; the curriculum controller stages a single real dropped log BEFORE
each episode. No scripted movement, command, target position or privileged
environment answer is sent to the neural actor DURING its scored actions.

This is isolated skill practice, not unscripted natural survival or Any%.
Both frozen eval and training require the same exact memory sensor.
"""
import argparse
from collections import Counter
import fcntl
import json
from math import hypot
import os
from pathlib import Path
import re
import signal
import sys
import time

from mcio_ctrl import controller,instance,types,world
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.agent_interface import AgentAction,MCioAgentBridge
from gmaincraft.instance_options import temporary_game_options
from gmaincraft.mcio_visible_items import ENV_NAME as ITEM_ENV, ON_DEMAND_ENV
from gmaincraft.numeric_item_pickup import (
    NumericPickupActor,VisiblePickupMemory,motor_action,ACTIONS,SCHEMA)
from gmaincraft.numeric_item_pickup_v2 import (
    SCHEMA as V2_SCHEMA, CASE_SCHEMA as V2_CASE,
    REWARD_SCHEMA as V2_REWARD, EgocentricPickupMemory, ReplayPickupQ,
    directional_cases_for_seed, progress_reward)
from gmaincraft.vision_settings import ensure_clean_visual_options
from log_interaction_probe import Client
from log_pickup_learner import DropArena,cases_for_seed

MCIO=ROOT/'mcio-data'
INSTANCE=MCIO/'instances'/'DemoInstance'
OPTIONS={'renderDistance':'4','enableVsync':'false','maxFps':'260'}
GAME_VERSION='1.21.3'
STOP=False


def on_stop(signum,frame):
    global STOP
    STOP=True


def run_paths(tag):
    return (ROOT/'checkpoints'/f'numeric_pickup_{tag}.pt',
            ROOT/'logs'/f'numeric_pickup_{tag}.jsonl',
            ROOT/'checkpoints'/f'numeric_pickup_{tag}.lock',
            INSTANCE/'saves'/f'MDL_NumericPickup_{tag}',
            ROOT/'logs'/f'numeric_pickup_{tag}.stop')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-tag',required=True)
    p.add_argument('--mode',choices=('train','eval'),default='train')
    p.add_argument('--kit',choices=('legacy_v1','egoreplay_v2'),default='legacy_v1')
    p.add_argument('--model-tag')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--episodes',type=int,default=60)
    p.add_argument('--max-actions',type=int,default=34)
    p.add_argument('--case-seed',type=int,default=221144)
    p.add_argument('--policy-seed',type=int,default=772211)
    p.add_argument('--case-profile',choices=('standard_v2','search_hard_v2'),default='standard_v2')
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='cpu')
    p.add_argument('--control',choices=('sampled','greedy','uniform_random','always_forward'),default='sampled')
    p.add_argument('--visible',action='store_true')
    p.add_argument('--report-every',type=int,default=10)
    args=p.parse_args(argv)
    if (not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,22}',args.run_tag) or
        (args.model_tag and not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,22}',args.model_tag))):
        p.error('Invalid run or model tag')
    if (not 1<=args.episodes<=50000 or not 5<=args.max_actions<=200 or
        not 1<=args.report_every<=10000 or not 1<=args.case_seed<2**63 or
        not 1<=args.policy_seed<2**63 or
        (args.mode=='train' and (args.model_tag or args.control!='sampled')) or
        (args.mode=='eval' and (not args.model_tag or args.resume)) or
        (args.kit=='legacy_v1' and args.case_profile!='standard_v2')):
        p.error('Invalid experiment budget, model contract or eval')
    return args


def policy_schema(args):
    return V2_SCHEMA if args.kit=='egoreplay_v2' else SCHEMA


def metadata(args):
    if args.kit=='egoreplay_v2':
        result=dict(schema=V2_SCHEMA,train_control='sampled',world=GAME_VERSION,
                    max_actions=args.max_actions,case_seed=args.case_seed,
                    policy_seed=args.policy_seed,
                    action_names=tuple(a[0] for a in ACTIONS),
                    observation='own_player+typed_visible_items_v1+egocentric_stale_memory',
                    case_contract=V2_CASE,reward_contract=V2_REWARD,
                    learner='persistent_real_transition_double_q_v2',
                    no_admin_commands_during_scored_play=True,
                    output_is_separate_skill_not_full_game=True,
                    memory_only_no_rgb=True)
        if args.case_profile!='standard_v2':
            result['case_profile']=args.case_profile
        return result
    return dict(schema=SCHEMA,train_control='sampled',world=GAME_VERSION,
                max_actions=args.max_actions,case_seed=args.case_seed,
                policy_seed=args.policy_seed,action_names=tuple(a[0] for a in ACTIONS),
                observation='own_player+typed_visible_items_v1+own_stale_memory',
                training_reward='held_inventory_gain+player_known_distance_progress+own_start_safety_v2',
                no_admin_commands_during_scored_play=True,
                output_is_separate_skill_not_full_game=True,
                memory_only_no_rgb=True)


def own_oak(player):
    return sum(slot.count for slot in player.main_inventory+player.offhand_inventory
               if slot.item_id=='minecraft:oak_log')


def publish_checkpoint(path,record):
    temp=path.with_suffix('.pt.pending')
    if temp.exists():raise FileExistsError('Old pending pickup checkpoint: inspect before resuming')
    with temp.open('xb') as out:
        torch.save(record,out)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp,path)


def journal_append(log,data):
    log.write(json.dumps(data,sort_keys=True)+'\n')
    log.flush()
    os.fsync(log.fileno())


def restore_checkpoint(path,log_path,args):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('schema')!=policy_schema(args) or saved.get('metadata')!=metadata(args):
        raise ValueError('Pickup checkpoint metadata or observation drift')
    rows=[json.loads(s) for s in log_path.read_text().splitlines() if s.strip()]
    committed=[x for x in rows if x.get('stage')=='episode_committed']
    if (any(e['episode']!=i for i,e in enumerate(committed)) or
        len(committed)>saved['next_episode']):
        raise ValueError('Pickup replay journal/episode progression mismatch')
    repair=None
    if saved['next_episode']==len(committed)+1:
        last=saved.get('latest_result')
        if not isinstance(last,dict) or last.get('episode')!=len(committed):
            raise ValueError('Cannot recover ambiguous checkpoint ahead of journal')
        repair=last
    elif saved['next_episode']!=len(committed):
        raise ValueError('Checkpoint cursor/journal disagreement')
    return saved,repair


def beyond_practice_boundary(start_xyz,current_xyz):
    """Fixture-only end condition using the player's OWN observed start/XYZ.

    The temporary arena is finite. Terminate an unsuccessful trajectory
    before the player falls off its bedrock floor, instead of issuing
    a privileged recovery/teleport during an active scored episode.
    """
    if len(start_xyz)!=3 or len(current_xyz)!=3:
        raise ValueError('Invalid own observed player coordinates')
    return (hypot(current_xyz[0]-start_xyz[0],
                  current_xyz[2]-start_xyz[2])>6.5 or
            current_xyz[1]<start_xyz[1]-3.)


def reset_case_safely(arena,case):
    """Only outside scored play: retry delayed server teleport at most twice."""
    for attempt in range(3):
        try:
            return arena.reset(case),attempt
        except RuntimeError as exc:
            if not any(msg in str(exc) for msg in ('Not teleported to fresh scenario', 'Stale item contaminated new episode')) or attempt==2:
                raise
            # A few integrated-server ticks may be needed after an
            # out-of-bounds previous life. No scripted *scored* movement.
            arena.client.scoring=False
            arena.client.advance(24)
    raise AssertionError('Unreachable retry exit')


def run_episode(arena,case,brain,args,episode):
    initial,fixture_retries=reset_case_safely(arena,case)
    bridge=MCioAgentBridge(arena.client.ctrl,initial)
    bridge.apply(AgentAction(ticks=1),request_visible_items=True,return_view=False)
    player=bridge.player_knowledge()
    sensor=bridge.visible_items()
    memory=(EgocentricPickupMemory() if args.kit=='egoreplay_v2'
            else VisiblePickupMemory())
    elapsed=0
    last_action=0
    reading=memory.observe(player,sensor,elapsed_ticks=elapsed,last_action=last_action)
    initial_held=own_oak(player)
    if initial_held:
        raise RuntimeError('Fixture contamination: already owned the target item')
    real_gains=0
    initially_visible=reading.item_seen
    starting_xyz=player.player_xyz
    out_of_bounds=False
    trajectory=[]
    reward_sum=0.
    seen_steps=0
    max_visible_item_candidates=reading.visible_candidate_count
    action_counts=Counter()
    acquired=False
    for step in range(args.max_actions):
        if STOP:break
        chosen=(1 if args.control=='always_forward' else
                brain.act(reading,greedy=args.control=='greedy',
                          uniform_random=args.control=='uniform_random'))
        action=motor_action(chosen)
        action_counts[ACTIONS[chosen][0]]+=1
        before=reading.distance_to_remembered
        bridge.apply(action,request_visible_items=True,return_view=False)
        elapsed+=action.ticks
        later_player=bridge.player_knowledge()
        later_sensor=bridge.visible_items()
        later=memory.observe(later_player,later_sensor,
                             elapsed_ticks=elapsed,last_action=chosen)
        seen_steps+=int(reading.item_seen)
        max_visible_item_candidates=max(
            max_visible_item_candidates,later.visible_candidate_count)
        after=later.distance_to_remembered
        # Reward only from own remembered current/previous camera sightings,
        # player position, own inventory and own health (never fixture target).
        count=own_oak(later_player)
        delta=max(0,count-initial_held-real_gains)
        real_gains+=delta
        acquired=bool(real_gains)
        out_of_bounds=beyond_practice_boundary(starting_xyz,later_player.player_xyz)
        failed=out_of_bounds or later_player.health<=0
        if args.kit=='egoreplay_v2':
            reward=progress_reward(before,after,action.ticks,
                                   inventory_gain=delta,failed=failed)
            terminal=bool(acquired or failed or step+1==args.max_actions)
            trajectory.append((reading.features,chosen,reward,action.ticks,
                               later.features,terminal))
        else:
            progress=max(-1.,min(1.,before-after)) if before is not None and after is not None else 0.
            reward=1.5*progress-.03*(action.ticks/6)+20.*delta-4.*int(failed)
            trajectory.append((reading.features,chosen,reward,action.ticks))
        reward_sum+=reward
        reading=later
        player=later_player
        if acquired or player.health<=0 or out_of_bounds:break
    train_metrics=brain.update(trajectory) if args.mode=='train' and trajectory else None
    result=dict(stage='episode_committed',episode=episode,case_seed=args.case_seed,
                case_dx=round(case.dx,3),case_dz=round(case.dz,3),
                case_yaw=round(case.yaw,3),scored_controller_commands=0,
                actual_acquired=bool(acquired),own_inventory_gain=real_gains,
                item_visible_at_first_action=bool(initially_visible),
                final_held_oak=own_oak(player),
                final_health=round(float(player.health),2),
                score_reward=round(reward_sum,4),decisions=len(trajectory),
                scored_game_ticks=elapsed,
                visible_decisions=seen_steps,
                max_visible_item_candidates=max_visible_item_candidates,
                fixture_reset_retries=fixture_retries,
                practice_out_of_bounds=bool(out_of_bounds),
                stopped_early=STOP and not acquired,
                actor_updates=brain.updates,
                action_counts=dict(action_counts),
                **({'update':train_metrics} if train_metrics else {}))
    print('NUMERIC PICKUP',json.dumps(result,sort_keys=True),flush=True)
    return result


def main(argv=None):
    args=parse_args(argv)
    checkpoint,journal,lock,game_save,stop_marker=run_paths(args.run_tag)
    model_checkpoint=ROOT/'checkpoints'/f'numeric_pickup_{args.model_tag}.pt' if args.model_tag else None
    if stop_marker.exists():raise FileExistsError('Stale stop marker; investigate earlier run')
    if args.resume and not (checkpoint.is_file() and journal.is_file() and game_save.is_dir()):
        raise FileNotFoundError('Cannot resume without full existing checkpoint, journal and game')
    if not args.resume and any(p.exists() for p in (journal,game_save,checkpoint)):
        raise FileExistsError('Refusing to overwrite prior pickup experiment')
    if args.mode=='eval':
        if not model_checkpoint.is_file():raise FileNotFoundError('Missing frozen pickup model')
        source=torch.load(model_checkpoint,map_location='cpu',weights_only=True)
        if source.get('schema')!=policy_schema(args) or args.case_seed==source['metadata']['case_seed']:
            raise ValueError('Frozen evaluation must use different fixture seed')
        if args.max_actions!=source['metadata']['max_actions']:
            raise ValueError('Frozen pickup motor horizon does not match source model')
    os.makedirs(journal.parent,exist_ok=True)
    with lock.open('a+') as held:
        try:fcntl.flock(held.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError('Already running: pickup advisory lock HELD') from e
        brain=(ReplayPickupQ(seed=args.policy_seed,device=args.device)
               if args.kit=='egoreplay_v2' else
               NumericPickupActor(seed=args.policy_seed,device=args.device))
        if args.kit=='egoreplay_v2':
            brain.training=args.mode=='train'
        cursor=0
        recovery=None
        if args.resume:
            saved,recovery=restore_checkpoint(checkpoint,journal,args)
            brain.restore(saved['brain'])
            cursor=saved['next_episode']
            if cursor>=args.episodes:raise ValueError('Already reached requested training episode target')
        elif args.mode=='eval':
            brain.restore(source['brain'],optimizer=False)
            # Evaluation stochastic controls use independent action RNG.
            if args.kit=='legacy_v1':
                torch.manual_seed(args.policy_seed)
        cases=(directional_cases_for_seed(args.case_seed,args.episodes,profile=args.case_profile)
               if args.kit=='egoreplay_v2' else
               cases_for_seed(args.case_seed,args.episodes))
        if not args.resume:
            world.WorldManager(mcio_dir=MCIO).copy('storage','DemoWorld',
                'DemoInstance',game_save.name,force=False)
        ensure_clean_visual_options(INSTANCE)
        env_extra={ITEM_ENV:'1',ON_DEMAND_ENV:'1',
                   'GMAINCRAFT_MEMORY_ONLY_FRAME_V1':'1',
                   '__GL_THREADED_OPTIMIZATIONS':'0'}
        options=types.RunOptions(mcio_dir=MCIO,instance_name='DemoInstance',
            world_name=game_save.name,mcio_mode=types.MCioMode.SYNC,
            hide_window=not args.visible,gpu_lib=None if args.visible else 'nvidia',
            width=320,height=180,env_extra=env_extra)
        launcher=instance.Launcher(options)
        ctrl=None
        signal.signal(signal.SIGTERM,on_stop)
        signal.signal(signal.SIGINT,on_stop)
        cumulative=Counter()
        started=time.perf_counter()
        with journal.open('a' if args.resume else 'x') as log:
            if not args.resume:
                journal_append(log,dict(stage='config',mode=args.mode,
                        metadata=metadata(args),
                        frozen_source=args.model_tag if args.mode=='eval' else None))
                if args.mode=='train':
                    # Valid start cursor for crash-before-first-episode recovery.
                    publish_checkpoint(checkpoint,dict(schema=policy_schema(args),
                        metadata=metadata(args),brain=brain.state(),
                        next_episode=0,latest_result=None))
            if recovery:
                journal_append(log,dict(recovery,stage='episode_committed',
                    recovered_from_committed_checkpoint=True))
            existing=[json.loads(s) for s in journal.read_text().splitlines() if s.strip()]
            for entry in existing:
                if entry.get('stage')=='episode_committed':
                    cumulative['episodes']+=1
                    cumulative['successes']+=int(entry['actual_acquired'])
                    cumulative['inventory_items']+=int(entry['own_inventory_gain'])
            try:
                with temporary_game_options(INSTANCE,OPTIONS,run_tag=args.run_tag):
                    launcher.launch(wait=False)
                    ctrl=controller.ControllerSync(action_port=options.action_port,
                        observation_port=options.observation_port,connection_timeout=70)
                    client=Client(ctrl)
                    initial=client.step(clear=True)
                    # A current memory-only transport with zero RGB is REQUIRED;
                    # we must not silently turn on images as a learned input.
                    if initial.frame_width!=0 or len(initial.frame)!=0:
                        raise RuntimeError('Expected explicit numeric-only Minecraft transport')
                    arena=DropArena(client,initial)
                    arena.prepare()
                    print('NUMERIC PICKUP READY',json.dumps(dict(tag=args.run_tag,
                        mode=args.mode,source=args.model_tag,cursor=cursor,
                        fixture=game_save.name,episodes=args.episodes,
                        memory_only=True,kit=args.kit,
                        actor_random_initialization=args.mode=='train')),
                        flush=True)
                    for ep in range(cursor,args.episodes):
                        if STOP or stop_marker.exists():break
                        result=run_episode(arena,cases[ep],brain,args,ep)
                        if args.mode=='train':
                            publish_checkpoint(checkpoint,dict(
                                schema=policy_schema(args),metadata=metadata(args),
                                brain=brain.state(),next_episode=ep+1,
                                latest_result=result))
                        journal_append(log,result)
                        cumulative['episodes']+=1
                        cumulative['successes']+=int(result['actual_acquired'])
                        cumulative['inventory_items']+=result['own_inventory_gain']
                        if (ep+1)%args.report_every==0:
                            print('NUMERIC PICKUP PULSE',json.dumps(dict(
                                episodes=cumulative['episodes'],
                                successes=cumulative['successes'],
                                items=cumulative['inventory_items'],
                                updates=brain.updates)),flush=True)
            finally:
                if ctrl is not None:
                    try:ctrl.send_stop()
                    except Exception as e:print('STOP WARNING',repr(e),flush=True)
                    ctrl.close()
                launcher.close()
            completed=cursor+(cumulative['episodes']-cursor)
            stage='completed' if cumulative['episodes']>=args.episodes else 'paused'
            summary=dict(stage=stage,episodes=cumulative['episodes'],
                         successes=cumulative['successes'],
                         inventory_items=cumulative['inventory_items'],
                         optimizer_updates=brain.updates if args.mode=='train' else 0,
                         source_model=args.model_tag,wall_s=round(time.perf_counter()-started,3))
            journal_append(log,summary)
            print('NUMERIC PICKUP SUMMARY',json.dumps(summary,sort_keys=True),flush=True)
            if stop_marker.exists():stop_marker.unlink()


if __name__=='__main__':main()
