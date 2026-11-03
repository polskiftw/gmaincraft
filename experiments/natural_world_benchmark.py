"""Natural-world FIRST-LOG transfer benchmark (not Any% or RL training).

One distinct seeded world template per seed, and a *fresh* copy of that
same world per diagnostic mode. No world editing, /tp or /give during play.
Policies get only rendered RGB pixels. The benchmark privately checks the
Minecraft inventory for natural wood, and counts synchronized game ticks.

Generation is an explicitly separate --prepare phase, so unexpected world
creation failures cannot silently switch to a copied flat test arena.
Use unique --run-tag. Existing worlds/logs/checkpoints are never overwritten.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from mcio_ctrl import controller, instance, types, world

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gmaincraft.agent_interface import MCioAgentBridge, natural_log_count
from gmaincraft.natural_eval import MODES, make_agent, plan_runs
from gmaincraft.vision_settings import ensure_clean_visual_options
from gmaincraft.instance_options import temporary_game_options

MCIO = ROOT / 'mcio-data'
INSTANCE = 'DemoInstance'


def parse_seeds(value):
    try:
        seeds = [int(t.strip()) for t in value.split(',')]
    except ValueError as e:
        raise argparse.ArgumentTypeError('Expected comma-separated integer world seeds') from e
    if not seeds:
        raise argparse.ArgumentTypeError('Supply at least one seed')
    return seeds


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag', required=True, help='fresh 1-24 ASCII identifier')
    ap.add_argument('--world-seeds', type=parse_seeds, required=True,
                    help='comma-separated REAL Minecraft world seeds')
    ap.add_argument('--modes', nargs='+', choices=MODES,
                    default=[mode for mode in MODES if mode not in ('memory_factorized','status_factorized')])
    ap.add_argument('--max-ticks', type=int, default=1200)
    ap.add_argument('--visible', action='store_true', help='use nested Gamescope on sandbox')
    ap.add_argument('--frame-width', type=int, default=320)
    ap.add_argument('--frame-height', type=int, default=180)
    ap.add_argument('--prepare', action='store_true',
                    help='generate independent seeded survival worlds ONLY, no game evaluation')
    ap.add_argument('--dry-run', action='store_true',
                    help='inspect world generation and evaluation plan without writing')
    ap.add_argument('--aim-model', default='rgb02')
    ap.add_argument('--pickup-model', default='p02')
    ap.add_argument('--memory-model',default=None,
                    help='local checkpoint tag joint_memory_<tag>.pt (V4)')
    ap.add_argument('--status-model',default=None,
                    help='opt-in patched-MCio checkpoint joint_status_<tag>.pt (V5)')
    ap.add_argument('--factorized-model',default=None,
                    help='local checkpoint tag joint_factorized_<tag>.pt')
    ap.add_argument('--joint-model', default=None,
                    help='local checkpoint tag joint_log_ppo_<tag>.pt')
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--action-seed', type=int, default=9971)
    ap.add_argument('--game-vsync', choices=('keep','on','off'), default='keep',
                    help='temporary MCio DemoInstance VSync override; restored after run')
    ap.add_argument('--game-fps', type=int, default=None,
                    help='temporary Minecraft maxFps, 10..260; restored after run')
    ap.add_argument('--render-distance', type=int, default=None,
                    help='temporary render distance 2..32; restored after run')
    ap.add_argument('--graphics-fast', action='store_true',
                    help='temporary Fast graphics: graphicsMode 0 (vanilla); restored')
    args = ap.parse_args(argv)
    if min(args.frame_width,args.frame_height)<96 or max(args.frame_width,args.frame_height)>1920:
        ap.error('Frame dimensions must be in the range 96..1920')
    if args.game_fps is not None and not 10<=args.game_fps<=260:
        ap.error('game-fps must be 10..260')
    if args.render_distance is not None and not 2<=args.render_distance<=32:
        ap.error('render-distance must be 2..32')
    if args.max_ticks < 1 or args.max_ticks > 72000:
        ap.error('max-ticks must be from 1 to 72000')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,23}', args.run_tag):
        ap.error('Invalid run tag')
    if len(args.world_seeds) != len(set(args.world_seeds)):
        ap.error('World seeds must be unique')
    if any(abs(s) >= 2**63 for s in args.world_seeds):
        ap.error('World seeds must fit signed 64-bit')
    if len(args.modes) != len(set(args.modes)):
        ap.error('Duplicate benchmark modes')
    if not all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}', t)
               for t in (args.aim_model, args.pickup_model, args.joint_model or 'unset',
                         args.factorized_model or 'unset', args.memory_model or 'unset', args.status_model or 'unset')):
        ap.error('Invalid checkpoint name')
    if 'joint_ppo' in args.modes and args.joint_model is None and not (args.dry_run or args.prepare):
        ap.error('--joint-model required for joint_ppo evaluation')
    if 'joint_factorized' in args.modes and args.factorized_model is None and not (args.dry_run or args.prepare):
        ap.error('--factorized-model required for joint_factorized evaluation')
    if 'memory_factorized' in args.modes and args.memory_model is None and not (args.dry_run or args.prepare):
        ap.error('--memory-model required for memory_factorized evaluation')
    if 'status_factorized' in args.modes and args.status_model is None and not (args.dry_run or args.prepare):
        ap.error('--status-model required for V5 status_factorized evaluation')
    return args


def paths_for(run):
    source = MCIO / 'world_storage' / run.source
    evaluated = MCIO / 'instances' / INSTANCE / 'saves' / run.world
    return source, evaluated


def load_weights(prefix, tag):
    path = ROOT / 'checkpoints' / f'{prefix}_{tag}.npz'
    with np.load(path) as data:
        return data['weights'].copy()


def print_plan(plans, args):
    print(f'NATURAL-WORLD FIRST-LOG BENCHMARK tag={args.run_tag} max_game_ticks={args.max_ticks}', flush=True)
    print('One generated vanilla-overworld seed per source; modes use fresh copies.', flush=True)
    print('Mode legacy_probe = old frozen toy policies + scripted handoff, NOT end-to-end RL.', flush=True)
    for p in plans:
        print(f'  seed={p.seed} source={p.source} eval={p.world} mode={p.mode}', flush=True)


def fix_spawn_and_verify(source, seed):
    """Verify real noise terrain / requested seed and disable spawn scatter."""
    from nbt import nbt
    level = source / 'level.dat'
    data = nbt.NBTFile(str(level))
    root = data['Data']
    settings = root['WorldGenSettings']
    if settings['seed'].value != seed:
        raise RuntimeError(f'Seed mismatch in {level}')
    generator = settings['dimensions']['minecraft:overworld']['generator']
    if generator['type'].value != 'minecraft:noise':
        raise RuntimeError(f'Not a natural noise-generated world: {level}')
    if 'spawnRadius' not in root['GameRules']:
        raise RuntimeError('No spawnRadius game rule found in level.dat')
    root['GameRules']['spawnRadius'].value = '0'
    data.write_file()
    # Re-read to ensure the source used for comparisons really has fixed spawn.
    check = nbt.NBTFile(str(level))['Data']
    if check['GameRules']['spawnRadius'].value != '0':
        raise RuntimeError('Could not fix spawn radius')


def prepare(plans):
    """WorldManager.create generates genuinely new worlds, not arena copies."""
    manager = world.WorldManager(mcio_dir=MCIO)
    for p in plans:
        if manager.world_exists('storage', p.source):
            raise FileExistsError(f'Source world already exists: {p.source} (not overwriting)')
    source_seeds = {p.source: p.seed for p in plans}
    for name, seed in source_seeds.items():
        print(f'GENERATE NORMAL SURVIVAL WORLD: {name} seed={seed}', flush=True)
        manager.create(name, mc_version='1.21.3', gamemode='survival',
                       difficulty='normal', seed=seed, reset_server_properties=True,
                       server_properties={'level-type': 'minecraft:normal',
                                          'generate-structures': 'true',
                                          'spawn-protection': '0',
                                          'online-mode': 'false',
                                          'server-ip': '127.0.0.1'})
        source = MCIO / 'world_storage' / name
        if not (source / 'level.dat').is_file():
            raise RuntimeError(f'Generation returned without level.dat: {name}')
        fix_spawn_and_verify(source, seed)
        print(f'PREPARED {name}: noise terrain, verified seed, fixed spawnRadius=0', flush=True)


def evaluate_one(p, args, weights, joint_checkpoint=None, factorized_checkpoint=None,
                 memory_checkpoint=None, status_checkpoint=None):
    manager = world.WorldManager(mcio_dir=MCIO)
    source, dest = paths_for(p)
    if not (source / 'level.dat').is_file():
        raise FileNotFoundError(f'Natural seed world missing {source}; run --prepare first')
    if dest.exists():
        raise FileExistsError(f'Evaluation world exists, refusing overwrite: {dest}')
    if p.mode == 'status_factorized':
        # Register the patched-mod CBOR option BEFORE the FIRST observation,
        # otherwise the decoder has no registered PlayerStatusOption subtype.
        from gmaincraft.mcio_status import PlayerStatusOption  # noqa: F401
    # Snapshot copied BEFORE running Minecraft, never changes the source save.
    manager.copy('storage', p.source, INSTANCE, p.world, force=False)
    opts = types.RunOptions(mcio_dir=MCIO, instance_name=INSTANCE, world_name=p.world,
                            mcio_mode=types.MCioMode.SYNC,
                            hide_window=not args.visible, gpu_lib=None if args.visible else 'nvidia',
                            width=args.frame_width, height=args.frame_height,
                            env_extra={'GMAINCRAFT_STATUS_V1':'1'} if
                                p.mode=='status_factorized' else {})
    ensure_clean_visual_options(MCIO/'instances'/INSTANCE)
    launch = instance.Launcher(opts)
    ctrl = None
    game_ticks = 0
    started = time.perf_counter()
    try:
        launch.launch(wait=False)
        ctrl = controller.ControllerSync(action_port=opts.action_port,
                                         observation_port=opts.observation_port,
                                         connection_timeout=100)
        ctrl.send_action(__import__('mcio_ctrl').network.ActionPacket(clear_input=True, commands=[]))
        initial = ctrl.recv_observation()
        if initial is None:
            raise RuntimeError('No initial Minecraft observation')
        bridge = MCioAgentBridge(ctrl, initial)
        if natural_log_count(bridge.packet) != 0:
            raise RuntimeError('Initial survival inventory already contains logs')
        start_pos = tuple(round(float(v), 5) for v in bridge.packet.player_pos)
        start_yaw = round(float(bridge.packet.player_yaw), 4)
        start_pitch = round(float(bridge.packet.player_pitch), 4)
        print(f'INITIAL {p.mode} seed={p.seed} pos={start_pos} yaw={start_yaw} pitch={start_pitch}', flush=True)
        policy = make_agent(p.mode, args.action_seed + p.seed,
                            aim_weights=weights[0], pickup_weights=weights[1],
                            joint_checkpoint=joint_checkpoint,
                            factorized_checkpoint=factorized_checkpoint,
                            memory_checkpoint=memory_checkpoint,
                            status_checkpoint=status_checkpoint,device=args.device)
        found = 0
        decisions = 0
        policy_wall = game_io_wall = observations_wall = 0.
        action_io_ms=[]
        while game_ticks < args.max_ticks and not found:
            clock = time.perf_counter()
            view=(bridge.multimodal_status_view() if p.mode=='status_factorized' else
                  bridge.multimodal_view() if p.mode=='memory_factorized' else bridge.view())
            observations_wall += time.perf_counter()-clock
            clock = time.perf_counter()
            action = policy.act(view, args.max_ticks - game_ticks)
            policy_wall += time.perf_counter()-clock
            clock = time.perf_counter()
            bridge.apply(action)
            elapsed_action=time.perf_counter()-clock
            game_io_wall += elapsed_action
            # Compact per-decision record, no world secrets/coordinates.
            action_io_ms.append([action.ticks,round(elapsed_action*1000/action.ticks,3)])
            decisions += 1
            game_ticks += action.ticks
            found = natural_log_count(bridge.packet)
            if bridge.packet.health <= 0:
                break
        return dict(seed=p.seed, mode=p.mode, result='success' if found else
                    'dead' if bridge.packet.health <= 0 else 'timeout',
                    found_logs=found, game_ticks=game_ticks,
                    decisions=decisions, wall_seconds=round(time.perf_counter() - started, 3),
                    game_io_wall_s=round(game_io_wall,4),
                    io_ms_per_tick=round(1000*game_io_wall/max(1,game_ticks),4),
                    policy_inference_wall_s=round(policy_wall,4),
                    observation_conversion_wall_s=round(observations_wall,4),
                    action_io_ms_per_tick=action_io_ms,
                    start_pos=start_pos, start_yaw=start_yaw, start_pitch=start_pitch,
                    frame_width=args.frame_width, frame_height=args.frame_height,
                    world=p.world, source=p.source)
    finally:
        if ctrl:
            try:
                ctrl.send_stop()
            except Exception as exc:
                print('MCio stop warning', repr(exc), flush=True)
            ctrl.close()
        launch.close()


def main(argv=None):
    args = parse_args(argv)
    plans = plan_runs(args.run_tag, args.world_seeds, args.modes)
    print_plan(plans, args)
    if args.dry_run:
        return
    if args.prepare:
        prepare(plans)
        return
    # Refuse partial runs: no existing evaluation worlds or output results.
    output = ROOT / 'logs' / f'natural_world_{args.run_tag}.jsonl'
    if output.exists():
        raise FileExistsError(f'Output exists: {output}')
    for p in plans:
        source, dest = paths_for(p)
        if not (source / 'level.dat').is_file():
            raise FileNotFoundError(f'Natural world missing: {source}; run --prepare')
        from nbt import nbt
        root = nbt.NBTFile(str(source / 'level.dat'))['Data']
        if (root['WorldGenSettings']['seed'].value != p.seed or
                root['GameRules']['spawnRadius'].value != '0' or
                root['WorldGenSettings']['dimensions']['minecraft:overworld']['generator']['type'].value != 'minecraft:noise'):
            raise RuntimeError('Source seed, natural terrain or deterministic spawn not verified')
        if dest.exists():
            raise FileExistsError(f'Existing eval world: {dest}')
    weights = (None, None)
    if 'legacy_probe' in args.modes:
        weights = (load_weights('log_aim_policy', args.aim_model),
                   load_weights('drop_pickup_policy', args.pickup_model))
    joint_checkpoint = None
    if 'joint_ppo' in args.modes:
        joint_checkpoint = ROOT / 'checkpoints' / f'joint_log_ppo_{args.joint_model}.pt'
        if not joint_checkpoint.is_file():
            raise FileNotFoundError(f'Joint PPO checkpoint unavailable: {joint_checkpoint}')
    status_checkpoint = None
    if 'status_factorized' in args.modes:
        status_checkpoint = ROOT / 'checkpoints' / f'joint_status_{args.status_model}.pt'
        if not status_checkpoint.is_file():
            raise FileNotFoundError(f'V5 status checkpoint unavailable: {status_checkpoint}')
    memory_checkpoint = None
    if 'memory_factorized' in args.modes:
        memory_checkpoint = ROOT / 'checkpoints' / f'joint_memory_{args.memory_model}.pt'
        if not memory_checkpoint.is_file():
            raise FileNotFoundError(f'Memory checkpoint unavailable: {memory_checkpoint}')
    factorized_checkpoint = None
    if 'joint_factorized' in args.modes:
        factorized_checkpoint = ROOT / 'checkpoints' / f'joint_factorized_{args.factorized_model}.pt'
        if not factorized_checkpoint.is_file():
            raise FileNotFoundError(f'Factorized checkpoint unavailable: {factorized_checkpoint}')
    output.parent.mkdir(parents=True, exist_ok=True)
    overrides={}
    if args.game_vsync!='keep':
        overrides['enableVsync']='true' if args.game_vsync=='on' else 'false'
    if args.game_fps is not None: overrides['maxFps']=str(args.game_fps)
    if args.render_distance is not None: overrides['renderDistance']=str(args.render_distance)
    if args.graphics_fast: overrides['graphicsMode']='0'
    print('TEMPORARY GAME OPTIONS:',overrides,'(original restored after run)',flush=True)
    starting_poses = {}
    with temporary_game_options(MCIO/'instances'/INSTANCE,overrides,run_tag=args.run_tag):
        with output.open('x', encoding='utf8') as file:
            for i, p in enumerate(plans, 1):
                try:
                    result = evaluate_one(p, args, weights, joint_checkpoint=joint_checkpoint,
                                          factorized_checkpoint=factorized_checkpoint,
                                          memory_checkpoint=memory_checkpoint,
                                          status_checkpoint=status_checkpoint)
                except Exception as e:
                    result = dict(seed=p.seed, mode=p.mode, result='error',
                                  error=f'{type(e).__name__}: {e}', world=p.world, source=p.source)
                    file.write(json.dumps(result) + '\n')
                    file.flush()
                    print(f'ERROR {i}/{len(plans)}: {result}', flush=True)
                    raise
                result['game_options']=dict(overrides)
                if p.mode in ('joint_factorized','joint_ppo','memory_factorized','status_factorized'):
                    import hashlib
                    path={'joint_factorized':factorized_checkpoint,'joint_ppo':joint_checkpoint,
                          'memory_factorized':memory_checkpoint,
                          'status_factorized':status_checkpoint}[p.mode]
                    result['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                pose = (tuple(result['start_pos']), result['start_yaw'], result['start_pitch'])
                if p.seed in starting_poses and starting_poses[p.seed] != pose:
                    mismatch = dict(seed=p.seed, mode=p.mode, result='invalid_spawn_mismatch',
                                    expected_pose=starting_poses[p.seed], actual_pose=pose,
                                    world=p.world, source=p.source)
                    file.write(json.dumps(mismatch) + '\n')
                    file.flush()
                    raise RuntimeError(f'Paired modes do not share identical initial pose: {mismatch}')
                starting_poses.setdefault(p.seed, pose)
                file.write(json.dumps(result) + '\n')
                file.flush()
                print(f'RUN {i}/{len(plans)}: {result}', flush=True)
    print('BENCHMARK COMPLETE. Natural log success requires observed inventory from normal Minecraft actions.', flush=True)
    print('Saved:', output, flush=True)


if __name__ == '__main__':
    main()
