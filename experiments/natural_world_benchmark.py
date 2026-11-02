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
    ap.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
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
    ap.add_argument('--joint-model', default=None,
                    help='local checkpoint tag joint_log_ppo_<tag>.pt')
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--action-seed', type=int, default=9971)
    args = ap.parse_args(argv)
    if min(args.frame_width,args.frame_height)<96 or max(args.frame_width,args.frame_height)>1920:
        ap.error('Frame dimensions must be in the range 96..1920')
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
               for t in (args.aim_model, args.pickup_model, args.joint_model or 'unset')):
        ap.error('Invalid checkpoint name')
    if 'joint_ppo' in args.modes and args.joint_model is None and not (args.dry_run or args.prepare):
        ap.error('--joint-model required for joint_ppo evaluation')
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


def evaluate_one(p, args, weights, joint_checkpoint=None):
    manager = world.WorldManager(mcio_dir=MCIO)
    source, dest = paths_for(p)
    if not (source / 'level.dat').is_file():
        raise FileNotFoundError(f'Natural seed world missing {source}; run --prepare first')
    if dest.exists():
        raise FileExistsError(f'Evaluation world exists, refusing overwrite: {dest}')
    # Snapshot copied BEFORE running Minecraft, never changes the source save.
    manager.copy('storage', p.source, INSTANCE, p.world, force=False)
    opts = types.RunOptions(mcio_dir=MCIO, instance_name=INSTANCE, world_name=p.world,
                            mcio_mode=types.MCioMode.SYNC,
                            hide_window=not args.visible, gpu_lib=None if args.visible else 'nvidia',
                            width=args.frame_width, height=args.frame_height)
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
                            joint_checkpoint=joint_checkpoint, device=args.device)
        found = 0
        decisions = 0
        while game_ticks < args.max_ticks and not found:
            view = bridge.view()
            action = policy.act(view, args.max_ticks - game_ticks)
            bridge.apply(action)
            decisions += 1
            game_ticks += action.ticks
            found = natural_log_count(bridge.packet)
            if bridge.packet.health <= 0:
                break
        return dict(seed=p.seed, mode=p.mode, result='success' if found else
                    'dead' if bridge.packet.health <= 0 else 'timeout',
                    found_logs=found, game_ticks=game_ticks,
                    decisions=decisions, wall_seconds=round(time.perf_counter() - started, 3),
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
    output.parent.mkdir(parents=True, exist_ok=True)
    starting_poses = {}
    with output.open('x', encoding='utf8') as file:
        for i, p in enumerate(plans, 1):
            try:
                result = evaluate_one(p, args, weights, joint_checkpoint=joint_checkpoint)
            except Exception as e:
                result = dict(seed=p.seed, mode=p.mode, result='error',
                              error=f'{type(e).__name__}: {e}', world=p.world, source=p.source)
                file.write(json.dumps(result) + '\n')
                file.flush()
                print(f'ERROR {i}/{len(plans)}: {result}', flush=True)
                raise
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
