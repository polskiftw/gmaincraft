"""Resumable scratch PPO and experimental off-policy Q on NEW survival seeds.

Each episode receives a fresh Minecraft noise-generated world (not one of a
small rotating cohort). The Minecraft seed is NEVER shown to the policy.
Training and evaluation seeds are disjoint by alternating INTERNAL permutation
indices; a keyed full-64-bit bijection provides nonrepeating signed seeds.

CURRENT BACKEND: one Minecraft client/integrated server per seed. MCio's
synchronized client/server semaphores have NO safe world-reload protocol yet;
DO NOT assume QuickPlay can switch without a JVM restart. Startup/worldgen
cost is measured per episode. A tested safe restart is preferable to a
client/server deadlock or falsely reported learned experience.

Crash contract: checkpoints only after Minecraft has saved/exited. A crashed
in-flight episode is quarantined, and its seed is retried from scratch using
the previous committed checkpoint. Partial journal rows are explicitly
uncommitted. Graceful stop commits even a partial episode as final and next
resume uses the NEXT fresh seed. No daemon, no arbitrary world deletions.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import random
import re
import signal
import sys
import time

from mcio_ctrl import controller,instance,network,types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.agent_interface import AgentAction,MCioAgentBridge
from gmaincraft.experience_archive import (
    EpisodeExperience,owned_inventory,archive_path,reconcile_uncommitted,compact_retention)
from gmaincraft.hindsight_skills import HindsightSkillModel,SCHEMA as HINDSIGHT_SCHEMA
from gmaincraft.replay_q_skill import ReplayQSkill,SCHEMA as REPLAY_Q_SCHEMA,CRACK_GOAL
from gmaincraft.replay_q_rare_v2 import RareOutcomeQSkill,SCHEMA as RARE_Q_V2_SCHEMA,WOOD_CRACK_GOAL
from gmaincraft.online_q_replay import update_from_player_replay
from gmaincraft.natural_pickup_transfer import (
    NaturalPickupTransfer,coarse_audit_heads,CONTRACT as PICKUP_TRANSFER_CONTRACT)
from gmaincraft.online_q_rare_v2 import learn_between_real_worlds
from gmaincraft.fresh_seed_schedule import SeedSchedule,SCHEMA as STREAM_SCHEMA
from gmaincraft.fresh_world_manager import FreshWorldManager,names
from gmaincraft.instance_options import temporary_game_options,restore_orphaned_game_options
from gmaincraft.mcio_process_watchdog import (WatchedControllerSync,MinecraftProcessExited,
                                               MinecraftObservationTimeout)
from gmaincraft.natural_online_policy import (NaturalPPO,NaturalTransition,SCHEMA,wood_id,
                          DECISION_TICKS,HEAD_SIZES,observe,action_from_heads,
                          PERSISTENCE_SCHEMA,PERSISTENCE_HEAD_SIZES,PERSISTENCE_DURATIONS,
                          action_from_persistence_heads,WOOD_ENGAGEMENT_SCHEMA,
                          MOTOR_V7_SCHEMA,MOTOR_V7_HEAD_SIZES,MOTOR_V7,
                          TEMPORAL_V9_SCHEMA,TEMPORAL_V9_SELF_FEATURES,
                          action_from_motor_v7,motor_v7_history,motor_v7_duration_index)
from gmaincraft.natural_online_session import NaturalSessionMemory
from gmaincraft.natural_diagnostics import NaturalActionAudit,WoodCrackProgress
from gmaincraft.visible_target_history import VisibleTargetHistory,SCHEMA as HISTORY_V9_SCHEMA,FEATURE_NAMES as HISTORY_V9_FEATURES
from gmaincraft.wood_target_reward import WoodTargetFirstContact,WoodEngagement,WoodSustainedAttack
from gmaincraft.vision_settings import ensure_clean_visual_options
from natural_online_train import (MCIO,INSTANCE,ENV_EXTRA,OVERRIDES,REWARDS,
                                   read_step,log_row)

RUN_TAG_RE=r'[A-Za-z0-9][A-Za-z0-9_-]{0,22}'
FRESH_REWARDS={**REWARDS,'wood_progress_v3':'wood_discovery_v2+nonrepeatable_own_wood_crack_stage_progress',
    'wood_persistence_v4':'wood_progress_v3+learned_normal_action_hold_duration_6_24_72_ticks',
    'wood_target_v5':'v4_motor+reduced_nonwood_curiosity+first_vanilla_crosshair_wood_contact+wood_progress_goal',
    'wood_engagement_v6':'v4_motor+player_visible_wood_features+one_time_crosshair_attack_engagement+actual_inventory_goal',
    'wood_motor_v7':'v5_rewards+v6_visible_wood+one_categorical_ordinary_motor_action_and_duration',
    'wood_sustain_v8':'v7_learned_motor+low_sighting_rewards+bounded_consecutive_stationary_wood_attack_reward',
    'wood_temporal_v9':'v8_exact_rewards_and_motor+own_visible_previous_target_crack_and_action_history'}


def single_motor_v7(args):
    return args.reward_profile in ('wood_motor_v7','wood_sustain_v8','wood_temporal_v9')


def variable_motor(args):
    return args.reward_profile in ('wood_persistence_v4','wood_target_v5','wood_engagement_v6','wood_motor_v7','wood_sustain_v8','wood_temporal_v9')

def policy_schema(args):
    return (TEMPORAL_V9_SCHEMA if args.reward_profile=='wood_temporal_v9' else
            MOTOR_V7_SCHEMA if single_motor_v7(args) else
            WOOD_ENGAGEMENT_SCHEMA if args.reward_profile=='wood_engagement_v6' else
            PERSISTENCE_SCHEMA if variable_motor(args) else SCHEMA)


def own_wood_held(player):
    """Player-visible inventory only, never server-secret mining/pickup stats."""
    if not hasattr(player,'main_inventory') or not hasattr(player,'offhand_inventory'):
        raise TypeError('Expected current typed own-player inventory')
    return sum(int(slot.count) for slot in player.main_inventory+player.offhand_inventory
               if wood_id(slot.item_id))


def event_reward_profile(profile):
    """V9 changes ONLY observed history, never V8's underlying reward.

    Preserve older legacy aliases exactly; full curriculum/reward changes
    must introduce a different new profile/checkpoint, not mutate V9.
    """
    if profile in ('wood_progress_v3','wood_persistence_v4'):
        return 'wood_discovery_v2'
    if profile=='wood_motor_v7':
        return 'wood_target_v5'
    if profile=='wood_temporal_v9':
        return 'wood_sustain_v8'
    return profile


def parse_args(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--mode',choices=('train','eval','collect'),default='train')
    ap.add_argument('--control',choices=('policy_sampled','policy_greedy','uniform_random',
                                         'untrained','goal_sampled','goal_greedy','q_sampled','q_greedy'),
                    default='policy_sampled')
    ap.add_argument('--model-run',help='train run whose frozen PPO checkpoint is evaluated')
    ap.add_argument('--goal-model',help='frozen hindsight skill tag (eval-only, not full-game planner)')
    ap.add_argument('--goal-item',help='one player-knowable item ID to acquire in skill-only evaluation')
    ap.add_argument('--q-model',help='frozen off-policy replay Q skill tag (eval-only)')
    ap.add_argument('--q-goal',help='the Q model training goal: observed inventory item or crack progress')
    ap.add_argument('--pickup-model',help='frozen player-visible pickup skill tag; eval-only Q transfer')
    ap.add_argument('--pickup-selector',choices=('shadow','visible_memory_v1'),default='shadow',
                    help='shadow measures natural drops without changing actions; visible_memory is an UNLEARNED player-only handoff diagnostic')
    ap.add_argument('--resume',action='store_true')
    cap=ap.add_mutually_exclusive_group()
    cap.add_argument('--decisions',type=int,default=None,help='TOTAL six-tick gameplay units (v4 may hold actions for multiple units), including resume')
    cap.add_argument('--until-stopped',action='store_true',help='train until stop file or Ctrl-C')
    ap.add_argument('--episode-decisions',type=int,default=2400,
                    help='maximum six-tick gameplay units per unique seed; v4 holds may span multiple units')
    ap.add_argument('--update-every',type=int,default=96)
    ap.add_argument('--report-every',type=int,default=60)
    ap.add_argument('--seed-root',type=int,default=917211,
                    help='deterministic fresh-seed stream (not a policy feature)')
    ap.add_argument('--policy-seed',type=int,default=731117)
    ap.add_argument('--q-updates-per-world',type=int,default=0,
                    help='collect-only real self-replay Q updates between saved Minecraft worlds')
    ap.add_argument('--q-replay-worlds',type=int,default=32,
                    help='bounded committed collection episodes replayed for Q updates')
    ap.add_argument('--q-replay-batch-size',type=int,default=48)
    ap.add_argument('--q-v2-anchor-weight',type=float,default=2.)
    ap.add_argument('--q-v2-success-weight',type=float,default=.6)
    ap.add_argument('--collect-seed-start',type=int,default=0,
                    help='collection-only index in the TRAIN seed stream; avoid Q source episodes')
    ap.add_argument('--eval-seed-start',type=int,default=0,
                    help='eval-only starting unique held-out seed index; never a policy feature')
    ap.add_argument('--reward-profile',choices=tuple(FRESH_REWARDS),default='wood_discovery_v2')
    ap.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    ap.add_argument('--retain-worlds',type=int,default=2,
                    help='keep this many most recent completed disposable worlds')
    ap.add_argument('--visible',action='store_true')
    ap.add_argument('--memory-only-no-rgb',action='store_true',
                    help='Experimental MCio: keep render/sync but omit glReadPixels and RGB transport; requires patched Fabric JAR')
    ap.add_argument('--record-experience',action='store_true',
                    help='Opt-in bounded player-knowable transition archive for later goal learning')
    ap.add_argument('--experience-limit-gb',type=int,default=20,
                    help='Per-run archive rolling disk cap in GiB; 0 preserves all episodes')
    ap.add_argument('--experience-max-episode-mb',type=int,default=256,
                    help='Max compressed archive size per one committed seed; fail closed if exceeded')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--slow-ms',type=float,default=12.)
    ap.add_argument('--max-crash-retries',type=int,default=2,
                    help='consecutive JVM/MCio crashes before exiting safely; not in learned policy')
    ap.add_argument('--nvidia-gl-threading',choices=('off','default'),default='off',
                    help='scoped Minecraft-only NVIDIA GL threaded optimization workaround; default off')
    args=ap.parse_args(argv)
    if not re.fullmatch(RUN_TAG_RE,args.run_tag) or (args.pickup_model and not re.fullmatch(RUN_TAG_RE,args.pickup_model)) or (args.model_run and not re.fullmatch(RUN_TAG_RE,args.model_run)) or (args.q_model and not re.fullmatch(RUN_TAG_RE,args.q_model)) or (
            args.goal_model and not re.fullmatch(RUN_TAG_RE,args.goal_model)):
        ap.error('Unsafe run/model tag')
    if args.decisions is None and not args.until_stopped:
        args.decisions=10000
    if (not 12<=args.episode_decisions<=100000 or not 8<=args.update_every<=512 or
        not 1<=args.report_every<=100000 or not 0<=args.retain_worlds<=10 or
        not 1<=args.slow_ms<=1000 or not 0<=args.max_crash_retries<=5 or
        not 0<=args.experience_limit_gb<=2048 or
        not 1<=args.experience_max_episode_mb<=2048 or
        not 0<=args.q_updates_per_world<=1000 or
        not 1<=args.q_replay_worlds<=128 or not 8<=args.q_replay_batch_size<=256 or
        (args.mode!='collect' and args.q_updates_per_world!=0) or
        not 0<=args.q_v2_anchor_weight<=10 or
        not 0<args.q_v2_success_weight<=10 or
        (args.decisions is not None and not 1<=args.decisions<=10**10)):
        ap.error('Invalid episode, update, retention, performance or decision bound')
    if (type(args.eval_seed_start) is not int or not 0<=args.eval_seed_start<2**63 or
        (args.mode!='eval' and args.eval_seed_start!=0) or
        (args.mode!='collect' and args.collect_seed_start!=0) or
        (args.mode=='collect' and not 0<=args.collect_seed_start<2**63)):
        ap.error('Held-out seed index must be 0 for training and within the eval stream')
    if args.mode=='train' and (args.model_run or args.goal_model or args.goal_item or args.q_model or args.q_goal or
                               args.control!='policy_sampled'):
        ap.error('Train has no model import/goal and uses sampled PPO actions')
    if args.mode=='eval' and args.record_experience:
        ap.error('Held-out evaluation can never be archived as training')
    if args.mode=='collect' and (
            args.control!='q_sampled' or not args.q_model or not args.q_goal or
            args.model_run or args.goal_model or args.goal_item or not args.record_experience):
        ap.error('Collection must use a frozen Q practice policy and record own experience')
    is_goal=args.control in ('goal_sampled','goal_greedy')
    is_q=args.control in ('q_sampled','q_greedy')
    if args.mode=='eval' and (args.resume or args.until_stopped or
                             (args.control.startswith('policy')!=(args.model_run is not None)) or
                             (is_goal!=(args.goal_model is not None)) or
                             (is_goal!=(args.goal_item is not None) and not is_q) or
                             (is_q!=(args.q_model is not None)) or
                             (is_q!=(args.q_goal is not None)) or
                             (not is_q and args.q_goal is not None)):
        ap.error('Evaluation is bounded; standard policy uses --model-run, '
                 'goal control uses --goal-model/--goal-item, Q control uses '
                 '--q-model/--q-goal and optional --goal-item score')
    if ((args.pickup_model and (args.mode!='eval' or not is_q or
                                args.q_goal!=WOOD_CRACK_GOAL)) or
        (not args.pickup_model and args.pickup_selector!='shadow')):
        ap.error('Frozen pickup transfer is eval-only with exact V9 Q wood goal; selector requires pickup model')
    if is_q and args.q_goal not in (CRACK_GOAL,WOOD_CRACK_GOAL):
        if not args.q_goal.startswith('minecraft:'):ap.error('Replay Q requires vanilla item goal')
        from gmaincraft.experience_archive import ObservedItemGoal
        try:ObservedItemGoal(args.q_goal)
        except ValueError as exc:ap.error(str(exc))
    if is_q and args.reward_profile!='wood_temporal_v9':
        ap.error('Replay Q requires exact V9 observation/action contract')
    if args.goal_item and (not args.goal_item.startswith('minecraft:') or
                           len(args.goal_item)>128 or not re.fullmatch(r'[a-z0-9_:./-]+',args.goal_item)):
        ap.error('Invalid observed-game-item goal ID')
    SeedSchedule(args.seed_root,'train')  # type + sign bound
    return args


def paths(tag,mode):
    stem='fresh_stream_'+tag
    checkpoint=ROOT/'checkpoints'/(stem+'.pt')
    journal=ROOT/'logs'/('fresh_stream_'+tag+'.jsonl' if mode in ('train','collect') else
                         'fresh_eval_'+tag+'.jsonl')
    inflight=ROOT/'logs'/('fresh_stream_'+tag+'.inflight.json')
    stop=ROOT/'logs'/('fresh_stream_'+tag+'.stop')
    lock=ROOT/'checkpoints'/('fresh_stream_'+tag+'.lock')
    return checkpoint,journal,inflight,stop,lock


def metadata(args):
    return dict(kind=('gmaincraft_q_collect_v2_rare_outcome' if args.q_goal==WOOD_CRACK_GOAL else
                      'gmaincraft_q_collect_v1') if args.mode=='collect' else
                'gmaincraft_fresh_stream_v2_fixed_spawn',policy_schema=policy_schema(args),
                **({'collector_q_model':args.q_model,'collector_q_goal':args.q_goal,
                    'collector_start_index':args.collect_seed_start,
                    'collector_q_updates_per_world':args.q_updates_per_world,
                    'collector_q_replay_worlds':args.q_replay_worlds,
                    'collector_q_replay_batch_size':args.q_replay_batch_size,
                    **({'collector_q_v2_anchor_weight':args.q_v2_anchor_weight,
                        'collector_q_v2_success_weight':args.q_v2_success_weight}
                       if args.q_goal==WOOD_CRACK_GOAL else {})} if args.mode=='collect' else {}),
                **({'duration_ticks':PERSISTENCE_DURATIONS,'horizon_unit_ticks':DECISION_TICKS} if variable_motor(args) else {}),
                **({'motor_v7_actions':MOTOR_V7,'policy_action_heads':MOTOR_V7_HEAD_SIZES} if single_motor_v7(args) else {}),
                **({'visible_target_history_schema':HISTORY_V9_SCHEMA,
                    'visible_target_history_features':HISTORY_V9_FEATURES} if args.reward_profile=='wood_temporal_v9' else {}),
                seed_schedule=STREAM_SCHEMA,seed_root=args.seed_root,
                episode_decisions=args.episode_decisions,
                update_every=args.update_every,
                reward_profile=args.reward_profile,
                policy_seed=args.policy_seed,settings=dict(OVERRIDES),
                game_version='1.21.3',spawn_radius=0,gameplay_commands=0,
                **({'frame_transport':'memory_only_frame_callback_v1'} if args.memory_only_no_rgb else {}),
                **({'pickup_model':args.pickup_model,'pickup_selector':args.pickup_selector,
                    'pickup_transfer_contract':PICKUP_TRANSFER_CONTRACT}
                   if args.pickup_model else {}),
                source='minecraft:noise',
                no_privileged_seed_policy_input=True)


def save_json_atomic(path,data):
    temp=path.with_suffix(path.suffix+'.pending')
    if temp.exists():raise FileExistsError('Unfinished atomic staging file requires inspection: '+str(temp))
    try:
        with temp.open('x',encoding='utf8') as stream:
            json.dump(data,stream,sort_keys=True)
            stream.write('\n')
            stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


def save_model(path,model,meta,cursor,decisions,totals):
    # Model/optimizer and all PyTorch CPU/CUDA RNG state are committed
    # together only after a verified Minecraft shutdown. This checkpoint
    # starts the NEXT fresh episode, never an ambiguous partially saved one.
    data=dict(schema=model.schema,stream=STREAM_SCHEMA,metadata=meta,
              model=model.state(),next_episode=int(cursor),decisions=int(decisions),
              totals=dict(totals))
    temp=path.with_suffix('.pt.pending')
    if temp.exists():raise FileExistsError('Old incomplete checkpoint stage: '+str(temp))
    try:
        model.torch.save(data,temp)
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


def restore(path,model,meta):
    data=model.torch.load(path,map_location=model.device,weights_only=True)
    if data['schema']!=model.schema or data['stream']!=STREAM_SCHEMA or data['metadata']!=meta:
        raise ValueError('Unsafe checkpoint drift; use same run settings, not a silently different experiment')
    model.restore(data['model'])
    return int(data['next_episode']),int(data['decisions']),Counter(data['totals'])


def quarantine_pending_file(path):
    """Preserve a crashed atomic-write stage, never overwrite it on resume."""
    path=Path(path)
    if not path.exists():return None
    for index in range(10000):
        target=path.with_name(path.name+f'.aborted_{index:04d}')
        if not target.exists():
            path.rename(target)
            return str(target)
    raise FileExistsError('Too many orphaned staging files: '+str(path))


def repair_jsonl_tail(path):
    """Make a crashed append-only journal safe for a fresh JSONL append.

    Never discard complete valid records, and never pretend that uncommitted
    rows were committed. Only a truncated final line can be quarantined.
    """
    path=Path(path)
    raw=path.read_bytes()
    prefix,sep,tail=raw.rpartition(b'\n')
    if not sep:
        raise ValueError('Missing full journal line; manual inspection required')
    for entry in (prefix+b'\n').splitlines():
        json.loads(entry.decode('utf8'))
    if not tail:return None
    try:
        json.loads(tail.decode('utf8'))
    except (ValueError,UnicodeDecodeError):
        for index in range(10000):
            quarantined=path.with_name(path.name+f'.partial_line_{index:04d}')
            if quarantined.exists():continue
            with quarantined.open('xb') as output:
                output.write(tail);output.flush();os.fsync(output.fileno())
            with path.open('r+b') as handle:
                handle.truncate(len(prefix)+1)
                handle.flush();os.fsync(handle.fileno())
            return dict(stage='recovered_partial_journal_line',
                        quarantined=str(quarantined),bytes_preserved=len(tail))
        raise FileExistsError('Too many quarantined journal tails')
    else:
        with path.open('ab') as handle:
            handle.write(b'\n');handle.flush();os.fsync(handle.fileno())
        return dict(stage='recovered_missing_journal_newline',bytes_preserved=len(tail))


def recover_inflight(marker,committed_episode,world_manager,log):
    if not marker.exists():return
    with marker.open(encoding='utf8') as handle:data=json.load(handle)
    index=data['episode']
    expected=SeedSchedule(data['seed_root'],data['namespace']).seed_at(index)
    if data['seed']!=expected or type(index) is not int:
        raise ValueError('In-flight episode marker was corrupted; refusing to alter world files')
    if index>committed_episode:
        raise ValueError('In-flight episode is ahead of saved model; manual recovery required')
    if index==committed_episode:
        moved=world_manager.quarantine(index)
        row=dict(stage='recovered_aborted_episode',episode=index,
                 seed=expected,quarantined=moved,
                 reason='uncommitted world changed after last model checkpoint; retry same seed')
    else:
        # Previous episode checkpoint committed but crash occurred before its
        # marker could be removed. Retire the completed owned world only.
        world_manager.retire(index,keep=False)
        row=dict(stage='recovered_stale_marker',episode=index,
                 seed=expected,reason='checkpoint already committed')
    log_row(log,row)
    print('FRESH RECOVERY',json.dumps(row),flush=True)
    marker.unlink()


def validate_memory_only_packet(packet):
    """Fail closed if the selected JAR did not disable the RGB payload."""
    if (packet.frame_width!=0 or packet.frame_height!=0 or len(packet.frame)!=0
            or packet.frame_sequence<=0):
        raise RuntimeError('Opt-in memory-only MCio returned RGB or missing frame tick: verify patched mod and launch environment')


def play_episode(args,model,manager,memory,episode,seed,budget,log,stop_requested,pickup_brain=None):
    """Run actual vanilla-survival gameplay; return uncommitted on-policy data.

    The caller commits a model checkpoint ONLY if this returns normally and
    the Minecraft launcher has fully closed its integrated server.
    """
    created=time.perf_counter()
    world_name=manager.prepare(episode,seed)
    worldgen_s=time.perf_counter()-created
    child_env={**ENV_EXTRA}
    if pickup_brain is not None:
        from gmaincraft.mcio_visible_items import ENV_NAME,ON_DEMAND_ENV
        child_env[ENV_NAME]='1'
        child_env[ON_DEMAND_ENV]='1'
    if args.memory_only_no_rgb:
        child_env['GMAINCRAFT_MEMORY_ONLY_FRAME_V1']='1'
    if args.nvidia_gl_threading=='off':
        child_env['__GL_THREADED_OPTIMIZATIONS']='0'
    options=types.RunOptions(mcio_dir=MCIO,instance_name='DemoInstance',
               world_name=world_name,mcio_mode=types.MCioMode.SYNC,
               hide_window=not args.visible,gpu_lib=None if args.visible else 'nvidia',
               width=320,height=180,env_extra=child_env,cleanup_on_signal=False)
    launcher=instance.Launcher(options)
    ctrl=None
    trajectory=[]
    events=Counter()
    audit=NaturalActionAudit()
    motor_v7_counts=Counter()
    motor_v7_on_wood_counts=Counter()
    crack_progress=WoodCrackProgress()
    target_history=VisibleTargetHistory() if args.reward_profile=='wood_temporal_v9' else None
    pickup=(NaturalPickupTransfer(pickup_brain,selector=args.pickup_selector)
            if pickup_brain is not None else None)
    pickup_elapsed_ticks=0
    target_reward=WoodTargetFirstContact(bonus=.3 if args.reward_profile in ('wood_sustain_v8','wood_temporal_v9') else .9) if args.reward_profile in ('wood_target_v5','wood_motor_v7','wood_sustain_v8','wood_temporal_v9') else None
    sustain_reward=WoodSustainedAttack() if args.reward_profile in ('wood_sustain_v8','wood_temporal_v9') else None
    engagement=WoodEngagement() if args.reward_profile=='wood_engagement_v6' else None
    elapsed_io=0.
    trained=0
    decisions=0  # Six-tick gameplay budget units; equal to game_ticks / 6.
    policy_actions=0  # Actual neural choices; v4 options may span 1, 4 or 12 budget units.
    last_report_bucket=0
    previous_frame=None
    experience=(EpisodeExperience(observation_schema=policy_schema(args),
        action_sizes=(MOTOR_V7_HEAD_SIZES if single_motor_v7(args) else
                      PERSISTENCE_HEAD_SIZES if variable_motor(args) else HEAD_SIZES))
                      if args.record_experience and args.mode in ('train','collect') else None)
    control_rng=random.Random(args.policy_seed ^ (episode*0x9E3779B97F4A7C15))
    launch_started=time.perf_counter()
    try:
        with temporary_game_options(INSTANCE,OVERRIDES,run_tag=args.run_tag):
            ensure_clean_visual_options(INSTANCE)
            launcher.launch(wait=False)
            ctrl=WatchedControllerSync(launcher=launcher,stop_requested=stop_requested,
                           action_port=options.action_port,
                           observation_port=options.observation_port,
                           connection_timeout=50,observation_timeout=30)
            initial_action=network.ActionPacket(clear_input=True,commands=[])
            ctrl.send_action(initial_action)
            initial=ctrl.recv_observation()
            if initial is None:
                raise RuntimeError('Minecraft returned no initial observation')
            bridge=MCioAgentBridge(ctrl,initial)
            if args.memory_only_no_rgb:
                validate_memory_only_packet(bridge.packet)
            memory.ensure_world(0,bridge.player_knowledge(),bridge._stats)
            bridge.apply(AgentAction(ticks=1),request_surface_objects=True,
                         request_mining_feedback=True,
                         request_visible_items=pickup is not None,return_view=False)
            bad_counter=[0]
            state,player,surfaces=read_step(bridge,memory,0,bad_counter=bad_counter,
                                                observation_schema=policy_schema(args),
                                                target_history=target_history)
            peak_own_wood_held=own_wood_held(player)
            first_player_pose=[round(float(x),4) for x in player.player_xyz]+[
                round(float(player.yaw_degrees),4),round(float(player.pitch_degrees),4)]
            if pickup is not None:
                pickup.observe(player,bridge.visible_items(),elapsed_ticks=0)
            client_startup_s=time.perf_counter()-launch_started
            goal_initial=(owned_inventory(player).get(args.goal_item,0)
                          if args.goal_item else 0)
            goal_max=goal_initial
            goal_previous=goal_initial
            print('FRESH WORLD READY',json.dumps(dict(episode=episode,seed=seed,
                    generator='minecraft:noise',world=world_name,
                    worldgen_s=round(worldgen_s,3),client_startup_s=round(client_startup_s,3),
                    max_decisions=budget,
                    nvidia_gl_threading=args.nvidia_gl_threading,
                    first_xyz=[round(float(x),2) for x in player.player_xyz])),flush=True)
            while decisions<budget:
                if stop_requested():break
                is_v4=variable_motor(args)
                action_sizes=(MOTOR_V7_HEAD_SIZES if single_motor_v7(args) else
                              PERSISTENCE_HEAD_SIZES if is_v4 else HEAD_SIZES)
                if args.mode=='eval' and args.control=='uniform_random':
                    heads=tuple(control_rng.randrange(n) for n in action_sizes)
                    logp=value=0.
                else:
                    heads,logp,value=model.act(state,greedy=(args.mode=='eval' and
                                         args.control in ('policy_greedy','goal_greedy','q_greedy')))
                action=(action_from_motor_v7(heads,(budget-decisions)*DECISION_TICKS)
                        if single_motor_v7(args) else
                        action_from_persistence_heads(heads,(budget-decisions)*DECISION_TICKS)
                        if is_v4 else action_from_heads(heads))
                pickup_action=(pickup.choose((budget-decisions)*DECISION_TICKS)
                               if pickup is not None else None)
                pickup_active=pickup_action is not None
                if pickup_active:action=pickup_action
                audit_heads=(coarse_audit_heads(action) if pickup_active else
                             motor_v7_history(heads) if single_motor_v7(args) else heads[:3])
                target_before=bridge.player_status().crosshair_block
                if pickup_active:
                    audit.observe_primitive(action,state,target_before)
                else:
                    audit.observe(audit_heads,state,target_before)
                    if single_motor_v7(args):
                        motor_v7_counts[MOTOR_V7[heads[0]][0]]+=1
                        if target_before is not None and wood_id(target_before.block_id):
                            motor_v7_on_wood_counts[MOTOR_V7[heads[0]][0]]+=1
                spent_ticks=0
                macro_reward=0.
                current_state=state
                current_player=player
                died=False
                # Long v4 actions are ONE neural selection, ordinary vanilla
                # held controls and read-only fresh observations every 6 ticks.
                # Intermediate samples are reward/evaluation-only, NOT an
                # opportunity for a scripted controller to choose a new action.
                while spent_ticks<action.ticks:
                    chunk_ticks=min(DECISION_TICKS,action.ticks-spent_ticks)
                    chunk=AgentAction(keys=action.keys,
                        look_dx=action.look_dx if spent_ticks==0 else 0.,
                        look_dy=action.look_dy if spent_ticks==0 else 0.,
                        ticks=chunk_ticks)
                    before=time.perf_counter()
                    bridge.apply(chunk,request_surface_objects=True,request_mining_feedback=True,
                                 request_visible_items=pickup is not None,return_view=False)
                    elapsed_io+=time.perf_counter()-before
                    current_frame=bridge.packet.frame_sequence
                    if previous_frame is not None and current_frame<=previous_frame:
                        raise AssertionError('Minecraft observation sequence stopped advancing')
                    previous_frame=current_frame
                    if args.memory_only_no_rgb:
                        validate_memory_only_packet(bridge.packet)
                    next_state,next_player,next_surfaces=read_step(bridge,memory,0,bad_counter=bad_counter,
                                                observation_schema=policy_schema(args),
                                                target_history=target_history,
                                                previous_action=chunk if target_history is not None else None,
                                                elapsed_action_ticks=chunk_ticks if target_history is not None else 0)
                    died=next_player.health<=0
                    if pickup is not None:
                        pickup_elapsed_ticks+=chunk_ticks
                        pickup.observe(next_player,bridge.visible_items(),
                                       elapsed_ticks=pickup_elapsed_ticks)
                    record=memory.accrue(0,next_player,next_surfaces,bridge._stats,audit_heads,
                                         chunk_ticks,death=died,
                                         reward_profile=event_reward_profile(args.reward_profile))
                    live_target=bridge.player_status().crosshair_block
                    progress_bonus=(crack_progress.observe(next_state,live_target)
                                    if args.reward_profile in ('wood_progress_v3','wood_persistence_v4','wood_target_v5','wood_engagement_v6','wood_motor_v7','wood_sustain_v8','wood_temporal_v9')
                                    else 0.)
                    # Same own visible crack progress; V5 pays twice the v3/v4
                    # amount to compete with long-hold exploration rewards.
                    if args.reward_profile in ('wood_target_v5','wood_motor_v7'):progress_bonus=round(progress_bonus*2,5)
                    if args.reward_profile in ('wood_engagement_v6','wood_sustain_v8','wood_temporal_v9'):progress_bonus=round(progress_bonus*3,5)
                    contact_bonus=target_reward.observe(live_target) if target_reward is not None else 0.
                    engagement_contact,engagement_attack=(
                        engagement.observe(live_target,attack=heads[2]==1)
                        if engagement is not None else (0.,0.))
                    sustain_bonus=(sustain_reward.observe(live_target,stationary_attack=(
                        audit_heads[2]==1 and not action.look_dx and not action.look_dy
                        and not any(k in action.keys for k in ('W','S','A','D','SPACE'))))
                        if sustain_reward is not None else 0.)
                    record['wood_crack_progress_reward']=progress_bonus
                    if target_reward is not None:record['wood_target_first_contact_reward']=contact_bonus
                    if sustain_reward is not None:record['wood_sustain_attack_reward']=sustain_bonus
                    if engagement is not None:
                        record['wood_engagement_contact_reward']=engagement_contact
                        record['wood_engagement_first_attack_reward']=engagement_attack
                    record['reward']=round(record['reward']+progress_bonus+contact_bonus+engagement_contact+engagement_attack+sustain_bonus,5)
                    if progress_bonus:events['wood_crack_progress_reward']+=progress_bonus
                    if sustain_bonus:events['wood_sustain_attack_reward']+=sustain_bonus
                    if contact_bonus:events['wood_target_first_contact_reward']+=contact_bonus
                    if engagement_contact:events['wood_engagement_contact_reward']+=engagement_contact
                    if engagement_attack:events['wood_engagement_first_attack_reward']+=engagement_attack
                    if args.q_goal==WOOD_CRACK_GOAL and (
                            next_state.own[TEMPORAL_V9_SELF_FEATURES.index('crosshair_is_wood')]>.5 and
                            next_state.own[TEMPORAL_V9_SELF_FEATURES.index('crack_stage_rose_same_target_div_10')]>0 and
                            next_state.own[TEMPORAL_V9_SELF_FEATURES.index('same_vanilla_crosshair_target_as_previous')]>.5):
                        events['v2_visible_wood_crack_rises']+=1
                    for key in ('fresh_cell','new_visible_surfaces','new_visible_wood',
                                'new_mined_blocks','new_wood_mined','wood_inventory_gain','death'):
                        events[key]+=record[key]
                    if pickup_active and record['wood_inventory_gain']:
                        events['own_log_gain_during_pickup_motor']+=record['wood_inventory_gain']
                    # Semi-Markov transition: discount intermediate rewards
                    # within one motor hold by their true game-tick offsets.
                    macro_reward+=(.9995**spent_ticks)*record['reward']
                    spent_ticks+=chunk_ticks
                    current_state=next_state
                    current_player=next_player
                    peak_own_wood_held=max(peak_own_wood_held,own_wood_held(current_player))
                    if record['new_mined_blocks'] or record['wood_inventory_gain'] or died:
                        log_row(log,dict(stage='uncommitted_game_event',episode=episode,
                             seed=seed,step=decisions+spent_ticks//DECISION_TICKS,**record))
                    if died or stop_requested():break
                audit.record_duration((0 if action.ticks<=6 else 1) if pickup_active else
                                      motor_v7_duration_index(heads) if single_motor_v7(args) else
                                      heads[3] if is_v4 else 0,spent_ticks,attack=audit_heads[2]==1,
                        wood_crosshair=target_before is not None and
                        wood_id(target_before.block_id))
                if args.mode=='train':
                    trajectory.append(NaturalTransition(state,heads,macro_reward,
                                         value,logp,spent_ticks,died))
                if args.goal_item:
                    # Scorer-only actual own inventory, never an oracle reward.
                    observed_count=owned_inventory(current_player).get(args.goal_item,0)
                    events['goal_item_gains']+=max(0,observed_count-goal_previous)
                    goal_max=max(goal_max,observed_count)
                    goal_previous=observed_count
                if experience is not None:
                    # Archive exactly the model's ordinary observations and
                    # the player's own inventory at BOTH ends of this action.
                    # No evaluator secrets, seed, target oracle or raw packet.
                    experience.add(state,heads,current_state,player,current_player,
                                   duration=spent_ticks,reward=macro_reward,done=died)
                decisions+=spent_ticks//DECISION_TICKS
                policy_actions+=1
                state=current_state
                player=current_player
                if args.mode=='train' and len(trajectory)>=args.update_every:
                    bootstrap=0. if died else model.act(state,greedy=True)[2]
                    metrics=model.update(trajectory,bootstrap_value=bootstrap)
                    trained+=1
                    trajectory.clear()
                    log_row(log,dict(stage='uncommitted_update',episode=episode,
                                 step=decisions,policy_actions=policy_actions,**metrics))
                report_bucket=decisions//args.report_every
                if report_bucket>last_report_bucket or decisions==budget:
                    last_report_bucket=report_bucket
                    pulse=dict(stage='uncommitted_pulse',episode=episode,seed=seed,
                               **({'motor_v7_counts':dict(motor_v7_counts),
                                   'motor_v7_on_wood_counts':dict(motor_v7_on_wood_counts)}
                                  if single_motor_v7(args) else {}),
                               decisions=decisions,policy_actions=policy_actions,
                               game_ticks=decisions*DECISION_TICKS,
                               game_io_s=round(elapsed_io,3),
                               game_io_ms_per_tick=round(1000*elapsed_io/(decisions*DECISION_TICKS),3),
                               totals=dict(events),action_audit=audit.summary(),
                               invalid_mining_readings=bad_counter[0],
                               **({'pickup_transfer_audit':pickup.audit()} if pickup else {}),
                               **({'visible_target_history_audit':target_history.summary()} if target_history else {}))
                    log_row(log,pulse)
                    print('FRESH PULSE',json.dumps(pulse),flush=True)
                    if (elapsed_io/(decisions*DECISION_TICKS))*1000>args.slow_ms:
                        print('FRESH PERFORMANCE WARNING: MCio tick slowdown',pulse['game_io_ms_per_tick'],flush=True)
            if trajectory and args.mode=='train':
                # A seed horizon is a terminal training boundary. Do not
                # bootstrap a new world using the value from the previous one.
                metrics=model.update(trajectory,bootstrap_value=0.)
                trained+=1
                trajectory.clear()
                log_row(log,dict(stage='uncommitted_update',episode=episode,
                                 step=decisions,**metrics))
    finally:
        try:
            if ctrl is not None:
                if launcher.poll() is None:
                    try:ctrl.send_stop()
                    except Exception as exc:print('STOP WARNING',repr(exc),flush=True)
                ctrl.close()
        finally:
            launcher.close()
    # No checkpoint before process exit/world save. If any exception above
    # escapes, its in-flight marker is left for quarantine on resume.
    from gmaincraft.fresh_world_manager import verify_noise_world
    verify_noise_world(manager.paths(episode)[1],seed,require_fixed_spawn=True)
    result=dict(episode=episode,seed=seed,decisions=decisions,policy_actions=policy_actions,
                first_player_pose=first_player_pose,
                **({'goal_item':args.goal_item,'goal_initial':goal_initial,
                    'goal_max_inventory':goal_max,
                    'goal_achieved':bool(goal_max>goal_initial)}
                   if args.goal_item else {}),
                **({'motor_v7_counts':dict(motor_v7_counts),
                    'motor_v7_on_wood_counts':dict(motor_v7_on_wood_counts)}
                   if single_motor_v7(args) else {}),
                game_ticks=decisions*DECISION_TICKS,
                updates=trained,worldgen_s=round(worldgen_s,3),
                client_startup_s=round(client_startup_s,3),
                game_io_s=round(elapsed_io,3),
                game_io_ms_per_tick=round(1000*elapsed_io/(max(1,decisions)*DECISION_TICKS),3),
                invalid_mining_readings=bad_counter[0],
                **({'pickup_transfer_audit':pickup.audit()} if pickup else {}),
                **({'visible_target_history_audit':target_history.summary()} if target_history else {}),
                counts=dict(events),action_audit=audit.summary(),
                final_own_wood_held=own_wood_held(player),
                peak_own_wood_held=peak_own_wood_held,
                final_own_health=round(float(player.health),3),
                stopped_early=decisions<budget)
    return result,experience



def load_frozen_pickup_control(root,args):
    """Require an unchanged real-practice model and frozen inference only."""
    if not args.pickup_model:return None
    import torch
    from gmaincraft.numeric_item_pickup_v2 import ReplayPickupQ,SCHEMA as PICKUP_SCHEMA
    from gmaincraft.numeric_item_pickup import ACTIONS
    path=root/'checkpoints'/f'numeric_pickup_{args.pickup_model}.pt'
    if not path.is_file():raise FileNotFoundError('Missing frozen numeric pickup checkpoint')
    state=torch.load(path,map_location='cpu',weights_only=True)
    meta=state.get('metadata',{})
    if (state.get('schema')!=PICKUP_SCHEMA or meta.get('schema')!=PICKUP_SCHEMA or
        tuple(meta.get('action_names',()))!=tuple(a[0] for a in ACTIONS) or
        state.get('brain',{}).get('schema')!=PICKUP_SCHEMA or
        state.get('next_episode',0)<20 or
        meta.get('world')!='1.21.3'):
        raise ValueError('Incompatible or untrained pickup checkpoint; cannot test natural-world transfer')
    # The original Q actor samples with PyTorch's global RNG. Loading this
    # second frozen network must NOT re-seed the mining actor and invalidate
    # same-seed sampled-control comparisons. Restore CPU+GPU RNG exactly.
    cpu_rng=torch.get_rng_state()
    cuda_rng=(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        pickup=ReplayPickupQ(seed=args.policy_seed ^ 0x54A15,device=args.device)
        pickup.restore(state['brain'],optimizer=False)
        pickup.training=False
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
    return pickup

class FrozenHindsightController:
    """Only a restricted goal-conditioned skill EVALUATOR, never a scripted plan."""
    updates=0

    def __init__(self,model,item_id):
        self.model=model
        self.item_id=item_id

    def act(self,observation,*,greedy=False):
        return self.model.act(observation,self.item_id,greedy=greedy),0.,0.


def load_goal_control(root,args):
    """Require exact source observation/action profile and disjoint seed root."""
    from gmaincraft.experience_archive import ObservedItemGoal
    ObservedItemGoal(args.goal_item)
    path=root/'checkpoints'/f'hindsight_{args.goal_model}.pt'
    if not path.is_file():raise FileNotFoundError('No trained hindsight skill checkpoint')
    import torch
    record=torch.load(path,map_location='cpu',weights_only=True)
    if record.get('schema')!=HINDSIGHT_SCHEMA:
        raise ValueError('Unknown versioned goal-policy checkpoint')
    meta=record['metadata']
    if (meta['source_observation_schema']!=policy_schema(args) or
        tuple(meta['action_sizes'])!=(MOTOR_V7_HEAD_SIZES if single_motor_v7(args)
                else PERSISTENCE_HEAD_SIZES if variable_motor(args) else HEAD_SIZES)):
        raise ValueError('Goal policy observation/action schema does not match evaluation')
    source_path=root/'checkpoints'/f"fresh_stream_{meta['source_run']}.pt"
    source=torch.load(source_path,map_location='cpu',weights_only=True)
    if (source['metadata']['seed_root']!=args.seed_root or
        source['metadata']['reward_profile']!=args.reward_profile or
        source['metadata']['policy_schema']!=meta['source_observation_schema']):
        raise ValueError('Goal policy trained on different source world-stream/motor/profile')
    # Frozen sampled policy RNG is an EVALUATION rollout setting, not the
    # original learner initialization seed. Restored weights are unchanged;
    # alternate seeds must produce independent actions on identical worlds.
    model=HindsightSkillModel(own_size=meta['own_size'],
                 action_sizes=tuple(meta['action_sizes']),
                 device=args.device,seed=args.policy_seed)
    model.restore(record['model'],optimizer=False)
    return FrozenHindsightController(model,args.goal_item)


class FrozenReplayQController:
    """A bounded Q practice policy; only collection mode permits updates."""

    def __init__(self,model,goal):
        self.model=model
        self.goal=goal
        self.torch=model.torch
        self.device=model.device
        # The outer controller, atomic checkpoint and inner Q model MUST have
        # the same version. V1 collectors retain their original schema.
        self.schema=RARE_Q_V2_SCHEMA if isinstance(model,RareOutcomeQSkill) else REPLAY_Q_SCHEMA

    @property
    def updates(self):
        return self.model.updates

    def act(self,observation,*,greedy=False):
        return self.model.act(observation,self.goal,greedy=greedy),0.,0.

    def state(self):
        return self.model.state()

    def restore(self,state):
        self.model.restore(state)
        return self


def load_replay_q_control(root,args):
    """Require unchanged V9 player sensors, source stream and action motor."""
    import torch
    record=torch.load(root/'checkpoints'/f'replay_q_{args.q_model}.pt',
                      map_location='cpu',weights_only=True)
    q_schema=record.get('schema')
    if q_schema not in (REPLAY_Q_SCHEMA,RARE_Q_V2_SCHEMA):
        raise ValueError('Unknown replay Q model schema')
    if q_schema==RARE_Q_V2_SCHEMA and args.q_goal!=WOOD_CRACK_GOAL:
        raise ValueError('V2 replay Q only supports its versioned observed wood crack practice goal')
    meta=record['metadata']
    if (meta.get('goal')!=args.q_goal or
        meta.get('source_observation_schema')!=TEMPORAL_V9_SCHEMA or
        tuple(meta.get('action_sizes',()))!=MOTOR_V7_HEAD_SIZES):
        raise ValueError('Replay Q goal/observation/motor contract mismatch')
    source=torch.load(root/'checkpoints'/f"fresh_stream_{meta['source_run']}.pt",
                      map_location='cpu',weights_only=True)
    if (source.get('stream')!=STREAM_SCHEMA or
        source['metadata']['seed_root']!=args.seed_root or
        source['metadata']['seed_root']!=meta['source_seed_root'] or
        source['metadata']['reward_profile']!='wood_temporal_v9' or
        source['metadata']['policy_schema']!=TEMPORAL_V9_SCHEMA or
        policy_schema(args)!=TEMPORAL_V9_SCHEMA or not single_motor_v7(args)):
        raise ValueError('Replay Q source/evaluation stream or player sensor drift')
    if q_schema==RARE_Q_V2_SCHEMA and args.mode=='collect' and (
            args.collect_seed_start<source['next_episode'] or
            meta['source_run']==args.run_tag):
        raise ValueError('V2 collection must use never-seen TRAIN indices after prior source cursor')
    model=(RareOutcomeQSkill if q_schema==RARE_Q_V2_SCHEMA else ReplayQSkill)(
          seed=args.policy_seed,device=args.device)
    model.restore(record['model'],optimizer=False)
    control=FrozenReplayQController(model,args.q_goal)
    if q_schema==RARE_Q_V2_SCHEMA:
        ancestral=meta.get('protected_prior_source_runs')
        if ancestral is None:
            ancestral=(meta.get('protected_prior_source_run',meta['source_run']),)
        if not isinstance(ancestral,(tuple,list)):
            raise ValueError('Invalid V2 immutable prior replay list')
        sources=tuple(ancestral)
        if meta['source_run'] not in sources:sources+=(meta['source_run'],)
        if (not sources or len(sources)>128 or len(sources)!=len(set(sources)) or
            any(not isinstance(tag,str) or not re.fullmatch(RUN_TAG_RE,tag)
                for tag in sources)):
            raise ValueError('Duplicate/invalid V2 prior source lineage')
        for protected in sources:
            prior=torch.load(root/'checkpoints'/f'fresh_stream_{protected}.pt',
                             map_location='cpu',weights_only=True)
            if prior.get('stream')!=STREAM_SCHEMA or (
                    prior['metadata']['seed_root']!=args.seed_root):
                raise ValueError('V2 protected historical replay uses different seed stream')
            if args.mode=='collect' and args.collect_seed_start<prior['next_episode']:
                raise ValueError('V2 cannot replay TRAIN seeds from protected historical source')
        control.protected_source_runs=sources
        control.prior_source_run=sources[0]
        control.additional_prior_source_run=sources[1] if len(sources)>1 else None
    else:
        control.protected_source_runs=()
        control.prior_source_run=None
        control.additional_prior_source_run=None
    return control


def main(argv=None):
    args=parse_args(argv)
    meta=metadata(args)
    schedule=SeedSchedule(args.seed_root,'train' if args.mode in ('train','collect') else 'eval')
    checkpoint,journal,inflight,stop_file,lock_path=paths(args.run_tag,args.mode)
    frozen_checkpoint=(ROOT/'checkpoints'/f'fresh_stream_{args.model_run}.pt'
                       if args.mode=='eval' and args.model_run else None)
    if args.dry_run:
        print('FRESH PLAN',json.dumps(dict(run=args.run_tag,mode=args.mode,
              options=vars(args),source='Minecraft noise server generation',
              seed_namespace=schedule.namespace,
              upcoming_seeds=[schedule.seed_at((args.collect_seed_start if args.mode=='collect'
                                      else args.eval_seed_start if args.mode=='eval' else 0)+i)
                              for i in range(5)],
              journal=str(journal),checkpoint=str(checkpoint),
              no_in_process_switch=True,
              note='Fresh Java integrated-server instance per episode; real benchmark will report restart overhead')),flush=True)
        return
    if args.mode=='eval':
        if journal.exists() or inflight.exists() or any((MCIO/'instances'/'DemoInstance'/'saves'/names(args.run_tag,i)[1]).exists() for i in range(2)):
            raise FileExistsError('Refusing to overwrite old evaluation world/report')
        if frozen_checkpoint is not None and not frozen_checkpoint.is_file():
            raise FileNotFoundError('Requested trained stream checkpoint not found')
    elif args.resume:
        if not checkpoint.is_file() or not journal.is_file():
            raise FileNotFoundError('No previous stream checkpoint/journal to resume')
    elif (checkpoint.exists() or journal.exists() or inflight.exists() or
          checkpoint.with_suffix('.pt.pending').exists() or
          inflight.with_suffix('.json.pending').exists()):
        raise FileExistsError('Existing training stream or unfinished stages; use --resume or inspect artifacts')
    model=(load_replay_q_control(ROOT,args) if args.mode in ('eval','collect') and args.q_model else
           load_goal_control(ROOT,args) if args.mode=='eval' and args.goal_model else
           NaturalPPO(seed=args.policy_seed,device=args.device,
                     learning_rate=.00015 if args.reward_profile in ('wood_progress_v3','wood_persistence_v4','wood_target_v5','wood_engagement_v6','wood_motor_v7','wood_sustain_v8','wood_temporal_v9') else .00035,
                     entropy_coef=.06 if args.reward_profile=='wood_engagement_v6' else
                                  .04 if args.reward_profile in ('wood_progress_v3','wood_persistence_v4','wood_target_v5','wood_motor_v7','wood_sustain_v8','wood_temporal_v9') else .014,
                     schema=policy_schema(args),
                     action_sizes=MOTOR_V7_HEAD_SIZES if single_motor_v7(args) else
                                  PERSISTENCE_HEAD_SIZES if variable_motor(args) else HEAD_SIZES))
    cursor=(args.eval_seed_start if args.mode=='eval' else
            args.collect_seed_start if args.mode=='collect' else 0)
    committed_decisions=0
    total=Counter()
    if args.mode in ('train','collect') and args.resume:
        cursor,committed_decisions,total=restore(checkpoint,model,meta)
    elif args.mode=='eval' and frozen_checkpoint:
        loaded=model.torch.load(frozen_checkpoint,map_location=model.device,weights_only=True)
        if loaded['stream']!=STREAM_SCHEMA or loaded['schema']!=model.schema:
            raise ValueError('Only checkpoints from the isolated fresh-seed stream can be frozen here')
        train_meta=loaded['metadata']
        if (train_meta.get('kind')!='gmaincraft_fresh_stream_v2_fixed_spawn' or
                train_meta['seed_root']!=args.seed_root or
                train_meta['seed_schedule']!=STREAM_SCHEMA):
            raise ValueError('Held-out test must use SAME stream root as trained model, other roots cannot guarantee disjoint seeds')
        if train_meta['reward_profile']!=args.reward_profile:
            raise ValueError('Reward version differs from frozen model metadata')
        model.restore(loaded['model'],optimizer=False)
    if args.decisions is not None and committed_decisions>=args.decisions:
        print('FRESH TARGET ALREADY COMPLETED',committed_decisions,flush=True)
        return
    pickup_brain=load_frozen_pickup_control(ROOT,args)
    manager=FreshWorldManager(MCIO,args.run_tag)
    stopped=[False]
    def on_stop(signum,_):
        stopped[0]=True
        print('FRESH STOP REQUESTED',signum,flush=True)
    signal.signal(signal.SIGINT,on_stop)
    signal.signal(signal.SIGTERM,on_stop)
    def stop_requested():
        if stop_file.is_file():stopped[0]=True
        return stopped[0]
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    journal.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a+') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Same seed-stream run lock is HELD. Another trainer '
                'or orphaned earlier sandbox may still be running. Do NOT delete '
                'the .lock file: it would permit concurrent checkpoint writes. '
                'Close the old game/trainer process or its isolated shell first.') from exc
        if args.resume and restore_orphaned_game_options(INSTANCE,OVERRIDES,run_tag=args.run_tag):
            print('FRESH RECOVERY restored interrupted disposable game options',flush=True)
        repaired=None
        abandoned_stages=[]
        if args.mode in ('train','collect') and args.resume:
            repaired=repair_jsonl_tail(journal)
            for stage_file in (checkpoint.with_suffix('.pt.pending'),
                               inflight.with_suffix('.json.pending')):
                saved=quarantine_pending_file(stage_file)
                if saved:abandoned_stages.append(saved)
        with journal.open('a' if args.resume else 'x',encoding='utf8') as log:
            if args.record_experience and args.mode in ('train','collect') and args.resume:
                orphaned=reconcile_uncommitted(ROOT/'data-local'/'experience',
                                               args.run_tag,cursor)
                if orphaned:
                    log_row(log,dict(stage='quarantined_uncommitted_experience',
                                     paths=orphaned,committed_cursor=cursor))
            if repaired:log_row(log,repaired)
            if abandoned_stages:
                log_row(log,dict(stage='recovered_pending_files',quarantined=abandoned_stages))
            if args.mode in ('train','collect') and not args.resume:
                # A collect stream may start AFTER the old V9 source seeds.
                # The initial committed cursor must match that FIRST seed,
                # otherwise startup crash recovery falsely sees a future seed.
                save_model(checkpoint,model,meta,cursor,0,total)
            if not args.resume:
                log_row(log,dict(stage='config',mode=args.mode,metadata=meta,
                      evaluation_control=args.control,
                      **({'q_model':args.q_model,'q_goal':args.q_goal,
                          'restricted_offpolicy_skill_evaluation':True}
                         if args.q_model else {}),
                      **({'goal_model':args.goal_model,'goal_item':args.goal_item,
                          'restricted_goal_conditioned_skill_evaluation':True}
                         if args.goal_model else {}),
                      evaluation_seed_start=args.eval_seed_start,
                      frozen_checkpoint=str(frozen_checkpoint) if frozen_checkpoint else None))
            if args.mode in ('train','collect') and args.resume:
                recover_inflight(inflight,cursor,manager,log)
            started=time.perf_counter()
            crash_streak=0
            while (args.decisions is None or committed_decisions<args.decisions) and not stop_requested():
                remaining=(args.decisions-committed_decisions if args.decisions is not None else args.episode_decisions)
                budget=min(args.episode_decisions,remaining)
                seed=schedule.seed_at(cursor)
                marker=dict(episode=cursor,seed=seed,seed_root=args.seed_root,
                            namespace=schedule.namespace,committed_decisions=committed_decisions)
                save_json_atomic(inflight,marker)
                try:
                    result,experience=play_episode(args,model,manager,NaturalSessionMemory(),cursor,
                                        seed,budget,log,stop_requested,pickup_brain=pickup_brain)
                except (MinecraftProcessExited,MinecraftObservationTimeout) as exc:
                    if args.mode not in ('train','collect'):raise
                    crash_streak+=1
                    reason=dict(stage='jvm_or_mcio_crash',episode=cursor,seed=seed,
                        committed_decisions=committed_decisions,attempt=crash_streak,
                        max_retries=args.max_crash_retries,exception=type(exc).__name__,
                        error=str(exc))
                    log_row(log,reason)
                    print('FRESH CRASH',json.dumps(reason),flush=True)
                    if crash_streak>args.max_crash_retries:
                        print('FRESH RETRY LIMIT reached; saved checkpoint is untouched',flush=True)
                        raise RuntimeError('Exceeded bounded Minecraft crash retries; '
                            'inspect JVM crash reports and resume later') from exc
                    # Discard *ALL* uncommitted neural updates and RNG
                    # from the interrupted episode, rather than train
                    # twice on any in-flight samples.
                    loaded_cursor,loaded_decisions,loaded_totals=restore(checkpoint,model,meta)
                    if loaded_cursor!=cursor or loaded_decisions!=committed_decisions:
                        raise RuntimeError('Checkpoint changed during crash retry')
                    total=loaded_totals
                    recover_inflight(inflight,cursor,manager,log)
                    continue
                crash_streak=0
                if experience is not None and result['policy_actions']:
                    # Archive FIRST: if interrupted before checkpoint commit,
                    # resume quarantines it by checkpoint cursor. Never learn
                    # from uncommitted episodes. World/JVM already closed.
                    archived=experience.save(ROOT/'data-local'/'experience',args.run_tag,
                        cursor,max_bytes=args.experience_max_episode_mb*(1024**2))
                    log_row(log,dict(stage='uncommitted_experience',episode=cursor,
                                      **archived))
                if args.mode=='collect':
                    # The latest episode has closed and its player-only archive
                    # was published, but no model/episode cursor has committed.
                    # A crash during Q updates reverts to the *previous* exact
                    # Q checkpoint and quarantines this episode on resume.
                    if args.q_goal==WOOD_CRACK_GOAL:
                        q_update=learn_between_real_worlds(model.model,
                            ROOT,args.run_tag,cursor,
                            protected_source_runs=model.protected_source_runs,
                            updates=args.q_updates_per_world,
                            batch_size=args.q_replay_batch_size,
                            recent_worlds=args.q_replay_worlds,
                            anchor_weight=args.q_v2_anchor_weight,
                            success_weight=args.q_v2_success_weight)
                    else:
                        q_update=update_from_player_replay(model.model,
                            ROOT/'data-local'/'experience',args.run_tag,cursor,
                            goal=args.q_goal,updates=args.q_updates_per_world,
                            batch_size=args.q_replay_batch_size,
                            max_episodes=args.q_replay_worlds)
                    log_row(log,dict(stage='uncommitted_online_q_update',
                                     episode=cursor,**q_update))
                    print('FRESH Q REPLAY UPDATE',json.dumps(q_update),flush=True)
                for key,value in result['counts'].items():total[key]+=value
                committed_decisions+=result['decisions']
                episode_completed=cursor
                cursor+=1
                if args.mode in ('train','collect'):
                    save_model(checkpoint,model,meta,cursor,committed_decisions,total)
                log_row(log,dict(stage='episode_committed',mode=args.mode,
                           next_episode=cursor,committed_decisions=committed_decisions,
                           **result))
                print('FRESH EPISODE COMMITTED',json.dumps(dict(
                      next_episode=cursor,total_decisions=committed_decisions,**result)),flush=True)
                if args.record_experience and args.experience_limit_gb:
                    pruned=compact_retention(ROOT/'data-local'/'experience',
                        args.run_tag,args.experience_limit_gb*(1024**3))
                    if pruned:
                        log_row(log,dict(stage='experience_retention_pruned',
                                          discarded=pruned,reason='opted_in_disk_cap'))
                # World cleanup is after the model checkpoint. Preserve the
                # most recent N completed owned worlds for user inspection.
                old=episode_completed-args.retain_worlds
                if old>=0:
                    manager.retire(old,keep=False)
                inflight.unlink()
                if args.mode in ('train','collect') and stop_requested():break
            summary=dict(stage='completed' if args.decisions is not None and
                         committed_decisions>=args.decisions else 'paused',
                         episodes=cursor,
                         evaluated_episodes=cursor-args.eval_seed_start if args.mode=='eval' else None,
                         evaluation_seed_start=args.eval_seed_start if args.mode=='eval' else None,
                         decisions=committed_decisions,
                         mode=args.mode,control=args.control,updates=model.updates,
                         seed_namespace=schedule.namespace,
                         new_optimizer_updates_in_eval=0 if args.mode=='eval' else None,
                         collector_q_model_optimizer_updates=model.model.updates if args.mode=='collect' else None,
                         totals=dict(total),wall_s=round(time.perf_counter()-started,2))
            log_row(log,summary)
            print('FRESH STREAM SUMMARY',json.dumps(summary),flush=True)
            print('FRESH STREAM SAVED',journal,checkpoint if args.mode in ('train','collect') else 'frozen/eval only',flush=True)
            if stop_file.is_file() and stopped[0]:stop_file.unlink()


if __name__=='__main__':main()
