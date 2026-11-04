"""Unscripted long-running, resumable, memory-first Minecraft survival PPO.

Rotates independently verified vanilla-noise survival save worlds; **no**
controller commands, placed blocks, pre-aimed logs, forced phases or pretrained
policies. Each world CONTINUES when revisited. The actor chooses movement,
look and attack together from its own live memory and first-hit camera
surfaces. Actual mined/picked-up events/inventory are outcome signals, while
ordinary exploration offers a dense but bounded early learning signal.

- train: scratch weights unless --resume, atomic checkpoint with optimizer,
  RNG, world-local explored map, visit counters and cumulative event counters.
- eval: separate NEVER-trained natural noise seeds, frozen checkpoint,
  identical observation encoder and action permissions.
- CLI exits cleanly at --decisions and handles SIGINT/SIGTERM after an action.
  Never deletes old copies/saves/models. Use GMAINCRAFT_GAMESCOPE_BACKEND=headless
  with tools/run-mcio-viewer.sh for unattended high-throughput RGB rendering.

Current visible-block input is provisional first-collision rays (13x9);
not a calibrated claim of perfect pixel-level perception. No item entity
observation exists yet. The first phase targets learning to *explore* real
worlds and discover resources without fixtures. No Any% promises.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
import random
from pathlib import Path
import re
import signal
import sys
import time

from mcio_ctrl import controller,instance,network,types,world
from nbt import nbt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.agent_interface import MCioAgentBridge,AgentAction
from gmaincraft.instance_options import temporary_game_options
from gmaincraft.mcio_mining_feedback import (MiningFeedbackOption,MiningFeedbackV1,
       ENV_NAME as MINING_ENV,ON_DEMAND_ENV as MINING_DEMAND_ENV)
from gmaincraft.mcio_surface_objects import (SurfaceObjectsV1,SurfaceObjectsOption,
       ENV_NAME as OBJECTS_ENV,ON_DEMAND_ENV as OBJECTS_DEMAND_ENV)
from gmaincraft.mcio_status import (PlayerStatusOption,ENV_NAME as STATUS_ENV)
from gmaincraft.natural_online_policy import (NaturalPPO,NaturalTransition,
       SCHEMA,DECISION_TICKS,HEAD_SIZES,observe,action_from_heads)
from gmaincraft.natural_online_session import NaturalSessionMemory
from gmaincraft.vision_settings import ensure_clean_visual_options

MCIO=ROOT/'mcio-data'
INSTANCE=MCIO/'instances'/'DemoInstance'
OVERRIDES={'renderDistance':'4','maxFps':'260','enableVsync':'false'}
ENV_EXTRA={STATUS_ENV:'1',OBJECTS_ENV:'1',OBJECTS_DEMAND_ENV:'1',
           MINING_ENV:'1',MINING_DEMAND_ENV:'1'}
REWARDS={'explore_v1':'novel_cells+first_hit_discovery+actual_mined_wood+inventory',
         'wood_discovery_v2':'novel_cells+first_hit_discovery+new_visible_wood+actual_mined_wood+inventory'}
NAME_PATTERN=r'[A-Za-z0-9][A-Za-z0-9_-]{0,28}'
SOURCE_PATTERN=r'[A-Za-z0-9][A-Za-z0-9_-]{0,74}'


def world_seed(name):
    path=MCIO/'world_storage'/name/'level.dat'
    if not path.is_file():raise FileNotFoundError(f'Missing prepared natural source world {path}')
    data=nbt.NBTFile(str(path))['Data']
    generator=data['WorldGenSettings']['dimensions']['minecraft:overworld']['generator']['type'].value
    if generator!='minecraft:noise':
        raise ValueError(f'Not a vanilla noise-generated survival world: {name} -> {generator}')
    if data['GameRules']['spawnRadius'].value!='0':
        raise ValueError(f'Expected fixed-zero spawn scatter for repeatable initial conditions: {name}')
    return int(data['WorldGenSettings']['seed'].value)


def parse_args(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag',required=True)
    world_args=ap.add_mutually_exclusive_group(required=True)
    world_args.add_argument('--sources',nargs='+',
                    help='verified independent natural-world names under mcio-data/world_storage')
    world_args.add_argument('--source-cohort',
                    help='tag from tools/prepare-natural-training-worlds.py; loads verified local manifest')
    ap.add_argument('--mode',choices=('train','eval'),default='train')
    ap.add_argument('--model-run',help='training tag supplying a frozen checkpoint in eval mode')
    ap.add_argument('--control',choices=('policy','policy_sampled','uniform_random','untrained'),default='policy',
                    help='frozen eval: greedy or sampled trained policy, random actions or scratch untrained model')
    ap.add_argument('--resume',action='store_true',
                    help='continue SAME run and evolving natural saves, not reset them')
    ap.add_argument('--decisions',type=int,default=20000,
                    help='TOTAL number of six-tick decisions including prior resumed progress')
    ap.add_argument('--switch-every',type=int,default=1200,
                    help='rotate to another existing evolving natural survival save')
    ap.add_argument('--update-every',type=int,default=96)
    ap.add_argument('--report-every',type=int,default=40)
    ap.add_argument('--reward-profile',choices=tuple(REWARDS),default='wood_discovery_v2',
                    help='versioned natural-only reward; choose explore_v1 for old checkpoint resume')
    ap.add_argument('--seed',type=int,default=730011,help='random-from-scratch policy seed')
    ap.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    ap.add_argument('--visible',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--slow-ms',type=float,default=12.,
                    help='warn if rolling game-IO time exceeds this many ms/tick')
    args=ap.parse_args(argv)
    if not re.fullmatch(NAME_PATTERN,args.run_tag):ap.error('Unsafe run tag')
    if args.model_run is not None and not re.fullmatch(NAME_PATTERN,args.model_run):
        ap.error('Unsafe model run name')
    if args.source_cohort:
        if not re.fullmatch(NAME_PATTERN,args.source_cohort):
            ap.error('Unsafe source cohort name')
        manifest=ROOT/'logs'/f'natural_sources_{args.source_cohort}.json'
        if not manifest.is_file():
            ap.error('Natural source manifest is missing: '+str(manifest))
        data=json.loads(manifest.read_text())
        if data.get('tag')!=args.source_cohort or data.get('world_generator')!='minecraft:noise':
            ap.error('Natural cohort manifest failed verification')
        args.sources=[entry['source'] for entry in data['sources']]
        for entry in data['sources']:
            if not re.fullmatch(SOURCE_PATTERN,entry['source']) or type(entry['seed']) is not int:
                ap.error('Malformed natural cohort entry')
            if world_seed(entry['source'])!=entry['seed']:
                ap.error('Natural cohort source seed mismatches actual level.dat')
    if any(not re.fullmatch(SOURCE_PATTERN,x) for x in args.sources):
        ap.error('Unsafe prepared natural source name')
    if (len(set(args.sources))!=len(args.sources) or not 1<=len(args.sources)<=16 or
        not 1<=args.decisions<=10_000_000 or not 10<=args.switch_every<=1000000 or
        not 8<=args.update_every<=512 or not 1<=args.report_every<=10000 or
        not .1<=args.slow_ms<=1000):
        ap.error('Invalid counts, performance bound or duplicate natural world')
    if args.mode=='eval' and ((args.control in ('policy','policy_sampled') and not args.model_run) or
                             (args.control not in ('policy','policy_sampled') and args.model_run) or args.resume):
        ap.error('Frozen policy eval needs --model-run; control modes use no pretrained checkpoint; no eval resume')
    if args.mode=='train' and (args.model_run or args.control!='policy'):
        ap.error('Training starts from scratch or --resume; controls only allowed in frozen eval')
    return args


def run_paths(tag,mode):
    identifier='natural_online_'+tag
    checkpoint=ROOT/'checkpoints'/f'{identifier}.pt'
    journal=ROOT/'logs'/f'{identifier}.jsonl' if mode=='train' else ROOT/'logs'/f'natural_frozen_{tag}.jsonl'
    return checkpoint,journal


def copied_world(tag,index,mode):
    return ('MDL_NaturalOnline_' if mode=='train' else 'MDL_NaturalFrozen_')+tag+f'_{index:02d}'


def prepare_worlds(tag,mode,sources):
    manager=world.WorldManager(mcio_dir=MCIO)
    for idx,name in enumerate(sources):
        dest=copied_world(tag,idx,mode)
        if (INSTANCE/'saves'/dest).exists():
            if mode=='eval':raise FileExistsError('Frozen evaluator refuses reused world: '+dest)
            continue  # existing evolving natural world; no reset, no overwrite
        manager.copy('storage',name,'DemoInstance',dest,force=False)
        print('NATURAL WORLD COPIED',dest,'from',name,flush=True)


def atomic_checkpoint(path,model,memory,metadata,decision,totals):
    torch=model.torch
    payload=dict(schema=SCHEMA,model=model.state(),memory=memory.state(),
                 metadata=metadata,decisions=int(decision),totals=dict(totals))
    tmp=path.with_name(path.name+'.pending')
    if tmp.exists():raise FileExistsError('Old incomplete checkpoint staging file; inspect first: '+str(tmp))
    try:
        torch.save(payload,tmp)
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def restored_state(path,model,metadata):
    data=model.torch.load(path,map_location=model.device,weights_only=True)
    if data['schema']!=SCHEMA:
        raise ValueError('Old/incompatible policy checkpoint schema')
    for key in ('sources','source_seeds','switch_every','settings','decision_ticks','seed','reward'):
        if data['metadata'][key]!=metadata[key]:
            raise ValueError(f'Unsafe resumed natural-world configuration drift: {key}')
    model.restore(data['model'])
    memory=NaturalSessionMemory().restore(data['memory'])
    return int(data['decisions']),Counter(data['totals']),memory


def fresh_memory(bridge,memory,index):
    player=bridge.player_knowledge()
    memory.ensure_world(index,player,bridge._stats)
    return player


def read_step(bridge,memory,index,*,bad_counter,observation_schema=SCHEMA):
    player=bridge.player_knowledge()
    status=bridge.player_status()
    surfaces=bridge.surface_objects()
    mining=None
    try:
        mining=bridge.mining_feedback()
    except ValueError as exc:
        # Actual game has rarely emitted an out-of-range crack stage while
        # exploring. Preserve STRICT adapter, record invalid raw data and
        # policy receives MISSING, not fabricated stage=0 or guessed value.
        raw=bridge.packet.get_option(MiningFeedbackOption)
        if raw is None:raise
        bad_counter[0]+=1
        print('INVALID MINING STAGE',json.dumps(dict(error=str(exc),
              raw_stage=raw.crack_stage,raw_active=raw.actively_destroying,
              packet_seq=bridge.packet.last_action_sequence)),flush=True)
    visit=memory.cell_visits(index,player)
    state=observe(player,status,surfaces,mining,start_xyz=memory.starts[index],
                  last_action=memory.last_actions[index],visit_count=visit,
                  elapsed_ticks=memory.ticks[index],schema=observation_schema)
    return state,player,surfaces


def log_row(stream,row):
    stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    stream.flush()


def main(argv=None):
    args=parse_args(argv)
    seeds=[world_seed(name) for name in args.sources]
    if len(set(seeds))!=len(seeds):
        raise ValueError('Different source names still had duplicate Minecraft seeds')
    meta=dict(schema=SCHEMA,run_tag=args.run_tag,sources=list(args.sources),source_seeds=seeds,
              switch_every=args.switch_every,settings=dict(OVERRIDES),
              decision_ticks=DECISION_TICKS,training_from='random_weights',
              observations='player_own_status+crosshair+bounded_13x9_first_hit_surfaces+own_crack',
              stage_changes_chosen_by_policy=True,no_admin_commands=True,
              reward=REWARDS[args.reward_profile],
              seed=args.seed)
    model_path,log_path=run_paths(args.run_tag,args.mode)
    checkpoint_for_eval=(ROOT/'checkpoints'/f'natural_online_{args.model_run}.pt'
                         if args.mode=='eval' and args.control in ('policy','policy_sampled') else
                         model_path if args.mode=='train' else None)
    if args.dry_run:
        print('NATURAL PLAN',json.dumps(dict(args=vars(args),metadata=meta,
              world_copies=[copied_world(args.run_tag,i,args.mode) for i in range(len(seeds))],
              checkpoint=str(checkpoint_for_eval),journal=str(log_path))),flush=True)
        return
    if args.mode=='eval':
        if checkpoint_for_eval is not None and not checkpoint_for_eval.is_file():
            raise FileNotFoundError('Missing trained model')
        if log_path.exists():raise FileExistsError('Evaluation log already exists')
    elif args.resume:
        if not model_path.exists() or not log_path.exists():
            raise FileNotFoundError('No resumable model+log from this run tag')
    elif model_path.exists() or log_path.exists() or any((INSTANCE/'saves'/copied_world(
            args.run_tag,i,args.mode)).exists() for i in range(len(seeds))):
        raise FileExistsError('Existing training artifact, use --resume (never overwrite)')
    model=NaturalPPO(seed=args.seed,device=args.device)
    memory=NaturalSessionMemory()
    totals=Counter()
    completed=0
    if args.mode=='eval':
        if args.control in ('policy','policy_sampled'):
            old=model.torch.load(checkpoint_for_eval,map_location=model.device,weights_only=True)
            trained_seeds=set(old['metadata']['source_seeds'])
            if set(seeds)&trained_seeds:
                raise ValueError('Evaluation seed overlaps training world; hold-out violated')
            if old['metadata']['reward']!=meta['reward']:
                raise ValueError('Use the same reward profile for frozen evaluation as training')
            model.restore(old['model'],optimizer=False)
            print('FROZEN policy loaded',checkpoint_for_eval,'updates',model.updates,flush=True)
        else:
            print('FROZEN untrained baseline',args.control,'seed',args.seed,flush=True)
    elif args.resume:
        completed,totals,memory=restored_state(model_path,model,meta)
        if completed>=args.decisions:
            print('TRAINING ALREADY REACHED decision cap',completed,flush=True)
            return
    # A kernel advisory lock prevents concurrent writers to the same run.
    stop_file=ROOT/'logs'/f'natural_online_{args.run_tag}.stop'
    lock_path=ROOT/'checkpoints'/f'natural_online_{args.run_tag}.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    log_path.parent.mkdir(parents=True,exist_ok=True)
    control_rng=random.Random(args.seed)
    with lock_path.open('a+') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Natural run is already active') from exc
        prepare_worlds(args.run_tag,args.mode,args.sources)
        stream_mode='a' if args.resume else 'x'
        stopped=[False]
        def signal_stop(signum,frame):
            stopped[0]=True
            print('REQUESTED GRACEFUL STOP',signum,flush=True)
        signal.signal(signal.SIGINT,signal_stop)
        signal.signal(signal.SIGTERM,signal_stop)
        with log_path.open(stream_mode,encoding='utf8') as log:
            if not args.resume:
                log_row(log,dict(stage='config',mode=args.mode,metadata=meta,
                         evaluation_control=args.control,
                         requested_decisions=args.decisions,
                         model_checkpoint=str(checkpoint_for_eval) if checkpoint_for_eval else None))
            clock=time.perf_counter()
            bad_counter=[0]
            while completed<args.decisions and not stopped[0]:
                index=(completed//args.switch_every)%len(args.sources)
                world_name=copied_world(args.run_tag,index,args.mode)
                end=min(args.decisions,((completed//args.switch_every)+1)*args.switch_every)
                options=types.RunOptions(mcio_dir=MCIO,instance_name='DemoInstance',
                         world_name=world_name,mcio_mode=types.MCioMode.SYNC,
                         hide_window=not args.visible,gpu_lib=None if args.visible else 'nvidia',
                         width=320,height=180,env_extra=ENV_EXTRA,cleanup_on_signal=False)
                launcher=instance.Launcher(options)
                ctrl=None
                trajectory=[]
                recent=Counter()
                last_io_ms=0.
                elapsed_io=0.
                launch_started=time.perf_counter()
                with temporary_game_options(INSTANCE,OVERRIDES,run_tag=args.run_tag):
                    ensure_clean_visual_options(INSTANCE)
                    try:
                        launcher.launch(wait=False)
                        ctrl=controller.ControllerSync(action_port=options.action_port,
                         observation_port=options.observation_port,connection_timeout=100)
                        packet=network.ActionPacket(clear_input=True,commands=[])
                        ctrl.send_action(packet)
                        initial=ctrl.recv_observation()
                        if initial is None:raise RuntimeError('Real Minecraft gave no initial packet')
                        bridge=MCioAgentBridge(ctrl,initial)
                        player=fresh_memory(bridge,memory,index)
                        bridge.apply(AgentAction(ticks=1),request_surface_objects=True,
                                     request_mining_feedback=True)
                        state,player,surface=read_step(bridge,memory,index,
                                                        bad_counter=bad_counter)
                        print('NATURAL WORLD LIVE',json.dumps(dict(source=args.sources[index],
                             seed=seeds[index],world=world_name,segment_end=end,
                             starting_xyz=[round(float(x),3) for x in player.player_xyz],
                             startup_s=round(time.perf_counter()-launch_started,2))),flush=True)
                        for _ in range(end-completed):
                            if stop_file.exists() and not stopped[0]:
                                stopped[0]=True
                                print('NATURAL GRACEFUL STOP FILE',stop_file,flush=True)
                            if stopped[0]:break
                            if args.mode=='eval' and args.control=='uniform_random':
                                actions=tuple(control_rng.randrange(n) for n in HEAD_SIZES)
                                logp=value=0.
                            else:
                                actions,logp,value=model.act(state,greedy=(args.mode=='eval' and args.control=='policy'))
                            action=action_from_heads(actions)
                            t0=time.perf_counter()
                            bridge.apply(action,request_surface_objects=True,
                                         request_mining_feedback=True)
                            io=time.perf_counter()-t0
                            elapsed_io+=io
                            last_io_ms=round(1000*io/action.ticks,3)
                            next_state,next_player,next_surface=read_step(bridge,memory,index,
                                                                         bad_counter=bad_counter)
                            dead=next_player.health<=0
                            record=memory.accrue(index,next_player,next_surface,
                                bridge._stats,actions,action.ticks,death=dead,
                                reward_profile=args.reward_profile)
                            recent.update({key:record[key] for key in (
                                'fresh_cell','new_visible_surfaces','new_visible_wood','new_mined_blocks',
                                'new_wood_mined','wood_inventory_gain','death')})
                            totals.update({key:record[key] for key in (
                                'fresh_cell','new_visible_surfaces','new_visible_wood','new_mined_blocks',
                                'new_wood_mined','wood_inventory_gain','death')})
                            if args.mode=='train':
                                trajectory.append(NaturalTransition(state,actions,
                                    record['reward'],value,logp,action.ticks,dead))
                            completed+=1
                            state=next_state
                            if record['new_mined_blocks'] or record['wood_inventory_gain'] or dead:
                                log_row(log,dict(stage='game_event',decision=completed,
                                     seed=seeds[index],world=index,**record))
                                print('NATURAL EVENT',json.dumps(dict(decision=completed,
                                    world=index,**record)),flush=True)
                            if args.mode=='train' and (len(trajectory)>=args.update_every or
                                    completed==end or stopped[0]):
                                bootstrap=(0. if dead else model.act(state,greedy=True)[2])
                                metrics=model.update(trajectory,bootstrap_value=bootstrap)
                                trajectory=[]
                                # Model updates are in RAM until the world has
                                # been shut down and saved normally. A neural
                                # checkpoint ahead of its *world save* could
                                # create inconsistent exploration credit after
                                # a power loss, so only commit at session end.
                                log_row(log,dict(stage='update',decision=completed,
                                     world=index,**metrics))
                            if (completed%args.report_every==0 or completed==end):
                                report=dict(stage='heartbeat',decisions=completed,
                                    world=index,seed=seeds[index],
                                    ticks=completed*DECISION_TICKS,
                                    io_ms_per_tick=last_io_ms,
                                    speed_ticks_per_s=round(6/max(io,1e-6),1),
                                    cumulative=dict(totals),recent=dict(recent),
                                    invalid_mining_readings=bad_counter[0],
                                    wall_s=round(time.perf_counter()-clock,1))
                                log_row(log,report)
                                print('NATURAL PULSE',json.dumps(report),flush=True)
                                if last_io_ms>args.slow_ms:
                                    print('NATURAL PERFORMANCE WARNING',last_io_ms,
                                          'ms/tick; inspect host load and render distance',flush=True)
                                recent.clear()
                    finally:
                        if ctrl is not None:
                            try:ctrl.send_stop()
                            except Exception as exc:print('STOP WARNING',repr(exc),flush=True)
                            ctrl.close()
                        launcher.close()
                if args.mode=='train':
                    # Save again AFTER Minecraft exited normally. This is a
                    # model+optimizer checkpoint, not a fabricated game reset.
                    atomic_checkpoint(model_path,model,memory,meta,completed,totals)
                log_row(log,dict(stage='world_paused',world=index,decisions=completed,
                                 elapsed_game_io_s=round(elapsed_io,3)))
            # Normal world shutdown already committed its final model and
            # memory; do not overwrite that checkpoint a second time.
            summary=dict(stage='completed' if completed>=args.decisions else 'paused',
                         mode=args.mode,evaluation_control=args.control,
                         decisions=completed,updates=model.updates,
                         cumulative=dict(totals),wall_s=round(time.perf_counter()-clock,2),
                         invalid_mining_readings=bad_counter[0])
            log_row(log,summary)
            print('NATURAL RUN SUMMARY',json.dumps(summary),flush=True)
            print('NATURAL SAVED',checkpoint_for_eval if args.mode=='eval' else model_path,
                  log_path,flush=True)
            if stopped[0] and stop_file.is_file():
                stop_file.unlink()  # completed user request, safe to resume later

if __name__=='__main__':main()
