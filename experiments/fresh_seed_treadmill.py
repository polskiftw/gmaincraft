"""Resumable from-scratch PPO on an unbounded stream of NEW survival seeds.

Each episode receives a fresh Minecraft noise-generated world (not one of a
small rotating cohort). The Minecraft seed is NEVER shown to the policy.
Training and evaluation seeds are mathematically disjoint by even/odd parity
of the full signed 64-bit seed; a keyed bijection provides nonrepeating seeds.

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
from gmaincraft.fresh_seed_schedule import SeedSchedule,SCHEMA as STREAM_SCHEMA
from gmaincraft.fresh_world_manager import FreshWorldManager,names
from gmaincraft.instance_options import temporary_game_options,restore_orphaned_game_options
from gmaincraft.mcio_process_watchdog import (WatchedControllerSync,MinecraftProcessExited,
                                               MinecraftObservationTimeout)
from gmaincraft.natural_online_policy import (NaturalPPO,NaturalTransition,SCHEMA,
                          DECISION_TICKS,HEAD_SIZES,observe,action_from_heads)
from gmaincraft.natural_online_session import NaturalSessionMemory
from gmaincraft.natural_diagnostics import NaturalActionAudit,WoodCrackProgress
from gmaincraft.vision_settings import ensure_clean_visual_options
from natural_online_train import (MCIO,INSTANCE,ENV_EXTRA,OVERRIDES,REWARDS,
                                   read_step,log_row)

RUN_TAG_RE=r'[A-Za-z0-9][A-Za-z0-9_-]{0,22}'
FRESH_REWARDS={**REWARDS,'wood_progress_v3':'wood_discovery_v2+nonrepeatable_own_wood_crack_stage_progress'}


def parse_args(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--mode',choices=('train','eval'),default='train')
    ap.add_argument('--control',choices=('policy_sampled','policy_greedy','uniform_random','untrained'),
                    default='policy_sampled')
    ap.add_argument('--model-run',help='train run whose frozen checkpoint is evaluated')
    ap.add_argument('--resume',action='store_true')
    cap=ap.add_mutually_exclusive_group()
    cap.add_argument('--decisions',type=int,default=None,help='TOTAL decisions, including resume')
    cap.add_argument('--until-stopped',action='store_true',help='train until stop file or Ctrl-C')
    ap.add_argument('--episode-decisions',type=int,default=2400,
                    help='maximum six-tick decisions spent per never-before-trained seed')
    ap.add_argument('--update-every',type=int,default=96)
    ap.add_argument('--report-every',type=int,default=60)
    ap.add_argument('--seed-root',type=int,default=917211,
                    help='deterministic fresh-seed stream (not a policy feature)')
    ap.add_argument('--policy-seed',type=int,default=731117)
    ap.add_argument('--reward-profile',choices=tuple(FRESH_REWARDS),default='wood_discovery_v2')
    ap.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    ap.add_argument('--retain-worlds',type=int,default=2,
                    help='keep this many most recent completed disposable worlds')
    ap.add_argument('--visible',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--slow-ms',type=float,default=12.)
    ap.add_argument('--max-crash-retries',type=int,default=2,
                    help='consecutive JVM/MCio crashes before exiting safely; not in learned policy')
    ap.add_argument('--nvidia-gl-threading',choices=('off','default'),default='off',
                    help='scoped Minecraft-only NVIDIA GL threaded optimization workaround; default off')
    args=ap.parse_args(argv)
    if not re.fullmatch(RUN_TAG_RE,args.run_tag) or (args.model_run and not re.fullmatch(RUN_TAG_RE,args.model_run)):
        ap.error('Unsafe run/model tag')
    if args.decisions is None and not args.until_stopped:
        args.decisions=10000
    if (not 12<=args.episode_decisions<=100000 or not 8<=args.update_every<=512 or
        not 1<=args.report_every<=100000 or not 0<=args.retain_worlds<=10 or
        not 1<=args.slow_ms<=1000 or not 0<=args.max_crash_retries<=5 or
        (args.decisions is not None and not 1<=args.decisions<=10**10)):
        ap.error('Invalid episode, update, retention, performance or decision bound')
    if args.mode=='train' and (args.model_run or args.control!='policy_sampled'):
        ap.error('Train has no model import and uses sampled PPO actions')
    if args.mode=='eval' and (args.resume or args.until_stopped or
                             (args.control.startswith('policy') and not args.model_run) or
                             (not args.control.startswith('policy') and args.model_run)):
        ap.error('Evaluation is bounded, non-resumable; policy controls require --model-run')
    SeedSchedule(args.seed_root,'train')  # type + sign bound
    return args


def paths(tag,mode):
    stem='fresh_stream_'+tag
    checkpoint=ROOT/'checkpoints'/(stem+'.pt')
    journal=ROOT/'logs'/('fresh_stream_'+tag+'.jsonl' if mode=='train' else
                         'fresh_eval_'+tag+'.jsonl')
    inflight=ROOT/'logs'/('fresh_stream_'+tag+'.inflight.json')
    stop=ROOT/'logs'/('fresh_stream_'+tag+'.stop')
    lock=ROOT/'checkpoints'/('fresh_stream_'+tag+'.lock')
    return checkpoint,journal,inflight,stop,lock


def metadata(args):
    return dict(kind='gmaincraft_fresh_stream_v2_fixed_spawn',policy_schema=SCHEMA,
                seed_schedule=STREAM_SCHEMA,seed_root=args.seed_root,
                episode_decisions=args.episode_decisions,
                update_every=args.update_every,
                reward_profile=args.reward_profile,
                policy_seed=args.policy_seed,settings=dict(OVERRIDES),
                game_version='1.21.3',spawn_radius=0,gameplay_commands=0,
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
    data=dict(schema=SCHEMA,stream=STREAM_SCHEMA,metadata=meta,
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
    if data['schema']!=SCHEMA or data['stream']!=STREAM_SCHEMA or data['metadata']!=meta:
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


def play_episode(args,model,manager,memory,episode,seed,budget,log,stop_requested):
    """Run actual vanilla-survival gameplay; return uncommitted on-policy data.

    The caller commits a model checkpoint ONLY if this returns normally and
    the Minecraft launcher has fully closed its integrated server.
    """
    created=time.perf_counter()
    world_name=manager.prepare(episode,seed)
    worldgen_s=time.perf_counter()-created
    child_env={**ENV_EXTRA}
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
    crack_progress=WoodCrackProgress()
    elapsed_io=0.
    trained=0
    decisions=0
    previous_frame=None
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
            memory.ensure_world(0,bridge.player_knowledge(),bridge._stats)
            bridge.apply(AgentAction(ticks=1),request_surface_objects=True,
                         request_mining_feedback=True)
            bad_counter=[0]
            state,player,surfaces=read_step(bridge,memory,0,bad_counter=bad_counter)
            client_startup_s=time.perf_counter()-launch_started
            print('FRESH WORLD READY',json.dumps(dict(episode=episode,seed=seed,
                    generator='minecraft:noise',world=world_name,
                    worldgen_s=round(worldgen_s,3),client_startup_s=round(client_startup_s,3),
                    max_decisions=budget,
                    nvidia_gl_threading=args.nvidia_gl_threading,
                    first_xyz=[round(float(x),2) for x in player.player_xyz])),flush=True)
            for i in range(budget):
                if stop_requested():break
                if args.mode=='eval' and args.control=='uniform_random':
                    heads=tuple(control_rng.randrange(n) for n in HEAD_SIZES)
                    logp=value=0.
                else:
                    heads,logp,value=model.act(state,greedy=(args.mode=='eval' and
                                                           args.control=='policy_greedy'))
                action=action_from_heads(heads)
                audit.observe(heads,state,bridge.player_status().crosshair_block)
                before=time.perf_counter()
                bridge.apply(action,request_surface_objects=True,request_mining_feedback=True)
                elapsed_io+=time.perf_counter()-before
                current_frame=bridge.packet.frame_sequence
                if previous_frame is not None and current_frame<=previous_frame:
                    raise AssertionError('Minecraft observation sequence stopped advancing')
                previous_frame=current_frame
                next_state,next_player,next_surfaces=read_step(bridge,memory,0,bad_counter=bad_counter)
                died=next_player.health<=0
                record=memory.accrue(0,next_player,next_surfaces,bridge._stats,heads,
                                     action.ticks,death=died,reward_profile=(
                                         'wood_discovery_v2' if args.reward_profile=='wood_progress_v3'
                                         else args.reward_profile))
                progress_bonus=(crack_progress.observe(next_state,bridge.player_status().crosshair_block)
                                if args.reward_profile=='wood_progress_v3' else 0.)
                record['wood_crack_progress_reward']=progress_bonus
                record['reward']=round(record['reward']+progress_bonus,5)
                if progress_bonus:events['wood_crack_progress_reward']+=progress_bonus
                for key in ('fresh_cell','new_visible_surfaces','new_visible_wood',
                            'new_mined_blocks','new_wood_mined','wood_inventory_gain','death'):
                    events[key]+=record[key]
                if args.mode=='train':
                    trajectory.append(NaturalTransition(state,heads,record['reward'],
                                         value,logp,action.ticks,died))
                decisions+=1
                state=next_state
                if record['new_mined_blocks'] or record['wood_inventory_gain'] or died:
                    log_row(log,dict(stage='uncommitted_game_event',episode=episode,
                         seed=seed,step=decisions,**record))
                if args.mode=='train' and len(trajectory)>=args.update_every:
                    bootstrap=0. if died else model.act(state,greedy=True)[2]
                    metrics=model.update(trajectory,bootstrap_value=bootstrap)
                    trained+=1
                    trajectory.clear()
                    log_row(log,dict(stage='uncommitted_update',episode=episode,
                                 step=decisions,**metrics))
                if decisions%args.report_every==0:
                    pulse=dict(stage='uncommitted_pulse',episode=episode,seed=seed,
                               decisions=decisions,game_ticks=decisions*DECISION_TICKS,
                               game_io_s=round(elapsed_io,3),
                               game_io_ms_per_tick=round(1000*elapsed_io/(decisions*DECISION_TICKS),3),
                               totals=dict(events),action_audit=audit.summary(),
                               invalid_mining_readings=bad_counter[0])
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
    result=dict(episode=episode,seed=seed,decisions=decisions,
                game_ticks=decisions*DECISION_TICKS,
                updates=trained,worldgen_s=round(worldgen_s,3),
                client_startup_s=round(client_startup_s,3),
                game_io_s=round(elapsed_io,3),
                game_io_ms_per_tick=round(1000*elapsed_io/(max(1,decisions)*DECISION_TICKS),3),
                invalid_mining_readings=bad_counter[0],
                counts=dict(events),action_audit=audit.summary(),
                stopped_early=decisions<budget)
    return result


def main(argv=None):
    args=parse_args(argv)
    meta=metadata(args)
    schedule=SeedSchedule(args.seed_root,'train' if args.mode=='train' else 'eval')
    checkpoint,journal,inflight,stop_file,lock_path=paths(args.run_tag,args.mode)
    frozen_checkpoint=(ROOT/'checkpoints'/f'fresh_stream_{args.model_run}.pt'
                       if args.mode=='eval' and args.model_run else None)
    if args.dry_run:
        print('FRESH PLAN',json.dumps(dict(run=args.run_tag,mode=args.mode,
              options=vars(args),source='Minecraft noise server generation',
              seed_namespace=schedule.namespace,
              upcoming_seeds=[schedule.seed_at(i) for i in range(5)],
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
    model=NaturalPPO(seed=args.policy_seed,device=args.device,
                     learning_rate=.00015 if args.reward_profile=='wood_progress_v3' else .00035,
                     entropy_coef=.04 if args.reward_profile=='wood_progress_v3' else .014)
    cursor=0
    committed_decisions=0
    total=Counter()
    if args.mode=='train' and args.resume:
        cursor,committed_decisions,total=restore(checkpoint,model,meta)
    elif args.mode=='eval' and frozen_checkpoint:
        loaded=model.torch.load(frozen_checkpoint,map_location=model.device,weights_only=True)
        if loaded['stream']!=STREAM_SCHEMA or loaded['schema']!=SCHEMA:
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
        if args.mode=='train' and args.resume:
            repaired=repair_jsonl_tail(journal)
            for stage_file in (checkpoint.with_suffix('.pt.pending'),
                               inflight.with_suffix('.json.pending')):
                saved=quarantine_pending_file(stage_file)
                if saved:abandoned_stages.append(saved)
        with journal.open('a' if args.resume else 'x',encoding='utf8') as log:
            if repaired:log_row(log,repaired)
            if abandoned_stages:
                log_row(log,dict(stage='recovered_pending_files',quarantined=abandoned_stages))
            if args.mode=='train' and not args.resume:
                save_model(checkpoint,model,meta,0,0,total)
            if not args.resume:
                log_row(log,dict(stage='config',mode=args.mode,metadata=meta,
                      evaluation_control=args.control,
                      frozen_checkpoint=str(frozen_checkpoint) if frozen_checkpoint else None))
            if args.mode=='train' and args.resume:
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
                    result=play_episode(args,model,manager,NaturalSessionMemory(),cursor,
                                        seed,budget,log,stop_requested)
                except (MinecraftProcessExited,MinecraftObservationTimeout) as exc:
                    if args.mode!='train':raise
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
                for key,value in result['counts'].items():total[key]+=value
                committed_decisions+=result['decisions']
                episode_completed=cursor
                cursor+=1
                if args.mode=='train':
                    save_model(checkpoint,model,meta,cursor,committed_decisions,total)
                log_row(log,dict(stage='episode_committed',mode=args.mode,
                           next_episode=cursor,committed_decisions=committed_decisions,
                           **result))
                print('FRESH EPISODE COMMITTED',json.dumps(dict(
                      next_episode=cursor,total_decisions=committed_decisions,**result)),flush=True)
                # World cleanup is after the model checkpoint. Preserve the
                # most recent N completed owned worlds for user inspection.
                old=episode_completed-args.retain_worlds
                if old>=0:
                    manager.retire(old,keep=False)
                inflight.unlink()
                if args.mode=='train' and stop_requested():break
            summary=dict(stage='completed' if args.decisions is not None and
                         committed_decisions>=args.decisions else 'paused',
                         episodes=cursor,decisions=committed_decisions,
                         mode=args.mode,control=args.control,updates=model.updates,
                         seed_namespace=schedule.namespace,
                         new_optimizer_updates_in_eval=0 if args.mode=='eval' else None,
                         totals=dict(total),wall_s=round(time.perf_counter()-started,2))
            log_row(log,summary)
            print('FRESH STREAM SUMMARY',json.dumps(summary),flush=True)
            print('FRESH STREAM SAVED',journal,checkpoint if args.mode=='train' else 'frozen/eval only',flush=True)
            if stop_file.is_file() and stopped[0]:stop_file.unlink()


if __name__=='__main__':main()
