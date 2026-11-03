"""Real Minecraft, no-Torch, synchronous MCio tick/GL pacing experiment.

Each unique-tag run copies the existing DemoWorld template; two repeats use the
same seeded, reset flat fixture and 300 identical normal vanilla input ticks.
No policy, torch, cheats during timed play, or Minecraft instance modification
outside an optional *temporarily restored* options.txt VSync/FPS override.
"""
import argparse
import json
import math
import re
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

from mcio_ctrl import controller, instance, network, types, world
from mcio_ctrl.envs import env_util, mcio_env

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gmaincraft.player_knowledge import PlayerKnowledgeV1
from gmaincraft.mcio_status import PlayerStatusV1, ENV_NAME
from gmaincraft.mcio_mining_feedback import MiningFeedbackV1, ENV_NAME as MINING_ENV
from gmaincraft.mcio_surface_grid import SurfaceGridV0, ENV_NAME as SURFACE_GRID_ENV
from gmaincraft.mcio_surface_objects import (SurfaceObjectsV1,SurfaceObjectsOption,SurfaceObjectsRequest,
                                          ENV_NAME as SURFACE_OBJECTS_ENV,ON_DEMAND_ENV)

ROOT = Path(__file__).resolve().parents[1]
MCIO = ROOT / 'mcio-data'
INSTANCE = MCIO / 'instances' / 'DemoInstance'


def action(ctrl, manager, keys=(), commands=(), clear=False, request_objects=False):
    states = {key: int(key in keys) for key in mcio_env.INPUT_MAP}
    events = manager.process_action(states, mcio_env.INPUT_MAP)
    pkt = network.ActionPacket(inputs=events, commands=list(commands), clear_input=clear,
                               options=[SurfaceObjectsRequest(True)] if request_objects else [])
    t0 = time.perf_counter_ns()
    ctrl.send_action(pkt)
    t1 = time.perf_counter_ns()
    obs = ctrl.recv_observation()
    t2 = time.perf_counter_ns()
    if obs is None or obs.version != network.MCIO_PROTOCOL_VERSION:
        raise RuntimeError('Invalid observation from game')
    if obs.last_action_sequence != pkt.sequence:
        raise RuntimeError(f'Out-of-sync action {pkt.sequence}, last={obs.last_action_sequence}')
    return obs, (t1-t0)/1e6, (t2-t1)/1e6


def setup(ctrl, fixed_xyz=None):
    manager = env_util.InputStateManager()
    obs, _, _ = action(ctrl, manager, clear=True)
    x, z = math.floor(obs.player_pos[0]), math.floor(obs.player_pos[2])
    y = math.floor(obs.player_pos[1]) - 1
    if fixed_xyz is not None:
        # Controller-only matching fixture; both A/B processes use the SAME
        # x/y/z instead of Minecraft's randomized safe spawn coordinates.
        x, y, z = fixed_xyz
    commands = [
        'difficulty peaceful', 'gamerule doMobSpawning false',
        'gamerule doDaylightCycle false', 'time set day', 'weather clear',
        f'fill {x-8} {y+1} {z-8} {x+8} {y+8} {z+8} minecraft:air replace',
        f'fill {x-8} {y} {z-8} {x+8} {y} {z+8} minecraft:bedrock replace',
        'gamemode survival @s',
    ]
    action(ctrl, manager, commands=commands, clear=True)
    for _ in range(4):
        obs, _, _ = action(ctrl, manager)
    return x, y, z


def reset(ctrl, xyz):
    x, y, z = xyz
    manager = env_util.InputStateManager()
    commands = [
        'kill @e[type=minecraft:item]',
        f'fill {x-5} {y+1} {z-5} {x+5} {y+5} {z+5} minecraft:air replace minecraft:oak_log',
        f'setblock {x} {y+1} {z+2} minecraft:oak_log replace',
        f'setblock {x} {y+2} {z+2} minecraft:oak_log replace',
        f'setblock {x} {y+3} {z+2} minecraft:oak_log replace',
        f'tp @s {x+0.5} {y+1.0} {z+0.5} 0 29',
        'clear @s', 'effect clear @s',
        'effect give @s minecraft:instant_health 1 5 true',
        'effect give @s minecraft:saturation 1 20 true',
    ]
    ob, _, _ = action(ctrl, manager, commands=commands, clear=True)
    for _ in range(40):
        ob, _, _ = action(ctrl, manager)
        wood = sum(int(item.count) for item in ob.inventory_main
                   if item.id == 'minecraft:oak_log')
        if (abs(ob.player_pos[0]-(x+0.5)) < .12 and
            abs(ob.player_pos[2]-(z+0.5)) < .12 and
            abs(ob.player_pitch-29) < 1 and wood == 0):
            break
    else:
        raise RuntimeError('Failed to settle fixture to identical pose and zero inventory')
    for _ in range(2):
        ob, _, _ = action(ctrl, manager)
    return manager


def do_run(ctrl, xyz, ticks, label, *, status_v1=False, surface_grid=False, surface_objects=False, on_demand=False, idle_only=False, mining_feedback=False):
    manager = reset(ctrl, xyz)
    stats = env_util.StatsCache()
    sends, receives, frames, stat_times, frame_times = [], [], [], [], []
    last_frame = -1
    frame_delta_mismatch = 0
    status_validations = 0
    feedback_validations = 0
    active_mining_packets = 0
    grid_validations = 0
    grid_ray_hits = 0
    object_validations = 0
    object_count = 0
    object_ray_coverage = 0
    start = time.perf_counter()
    for tick in range(ticks):
        keys = () if idle_only else ('LEFT_BUTTON',) if tick < 110 else ('W',) if tick < 145 else ()
        request = on_demand and (tick+1)%6==0
        ob, send_ms, recv_ms = action(ctrl, manager, keys=keys,request_objects=request)
        if on_demand and not request and ob.get_option(SurfaceObjectsOption) is not None:
            raise AssertionError('Unexpected unrequested surface data')
        sends.append(send_ms)
        receives.append(recv_ms)
        frame_delta_mismatch += int(last_frame >= 0 and ob.frame_sequence != last_frame+1)
        last_frame = ob.frame_sequence
        frames.append(len(ob.frame))
        t0 = time.perf_counter_ns()
        stats.update_cache(ob)
        knowledge = PlayerKnowledgeV1.from_packet(ob,stats=stats)
        if knowledge.player_xyz != tuple(ob.player_pos) or knowledge.health != ob.health:
            raise AssertionError('Player memory observation drift')
        if status_v1:
            status=PlayerStatusV1.from_packet(ob)
            if not 0<=status.hunger<=20:
                raise AssertionError('Invalid hunger status')
            status_validations+=1
        if mining_feedback:
            feedback=MiningFeedbackV1.from_packet(ob)
            feedback_validations+=1
            active_mining_packets+=int(feedback.actively_destroying)
        if surface_grid:
            grid=SurfaceGridV0.from_packet(ob)
            grid_ray_hits+=len(grid.hits)
            grid_validations+=1
        if surface_objects and (not on_demand or request):
            grouped=SurfaceObjectsV1.from_packet(ob)
            object_validations+=1
            object_count+=len(grouped.objects)
            object_ray_coverage+=sum(len(o.ray_indices) for o in grouped.objects)
        stat_times.append((time.perf_counter_ns()-t0)/1e6)
        if tick % 6 == 0:
            t0 = time.perf_counter_ns()
            img = ob.get_frame()
            if img.dtype.name != 'uint8' or img.shape != (ob.frame_height, ob.frame_width, 3):
                raise AssertionError('Invalid rendered RGB image')
            frame_times.append((time.perf_counter_ns()-t0)/1e6)
    elapsed = time.perf_counter() - start
    logs = sum(int(item.count) for item in ob.inventory_main if item.id == 'minecraft:oak_log')
    def describe(samples):
        values = sorted(samples)
        return dict(total_ms=round(sum(values),3), p50_ms=round(statistics.median(values),3),
                    p95_ms=round(values[int(.95*(len(values)-1))],3),
                    max_ms=round(values[-1],3))
    result = dict(label=label, ticks=ticks, action_profile='idle' if idle_only else 'attack_walk',
                  wall_s=round(elapsed,4),
                  ticks_per_wall_s=round(ticks/elapsed,2),
                  frame_size=[ob.frame_width, ob.frame_height], frame_bytes=round(statistics.median(frames)),
                  send=describe(sends), recv=describe(receives),
                  stats=describe(stat_times), rgb=describe(frame_times), rgb_samples=len(frame_times),
                  frame_sequence_gaps=frame_delta_mismatch, final_inventory_oak_logs=logs,
                  final_pos=[round(float(v),3) for v in ob.player_pos],
                  last_action_sequence=int(ob.last_action_sequence),
                  knowledge_schema=knowledge.schema, status_v1_checks=status_validations,
                  mining_feedback_checks=feedback_validations,
                  mining_active_packets=active_mining_packets,
                  surface_grid_checks=grid_validations, surface_grid_hits=grid_ray_hits,
                  surface_object_checks=object_validations, surface_object_records=object_count,
                  surface_object_rays=object_ray_coverage,
                  inventory_logs_v1=sum(
                      slot.count for slot in knowledge.main_inventory if slot.item_id=='minecraft:oak_log'))
    print('THROUGHPUT '+json.dumps(result,sort_keys=True),flush=True)
    return result


@contextmanager
def temporary_options(vsync, max_fps, tag):
    if vsync == 'keep' and max_fps is None:
        yield
        return
    file = INSTANCE / 'options.txt'
    backup = MCIO / f'options_throughput_restore_{tag}.bak'
    if backup.exists():
        raise RuntimeError(f'Backup already exists: {backup}; restore it before rerunning')
    original = file.read_bytes()
    with backup.open('xb') as out:
        out.write(original)
    try:
        lines = original.decode('utf-8').splitlines()
        replacements = {}
        if vsync != 'keep':
            replacements['enableVsync'] = 'true' if vsync == 'on' else 'false'
        if max_fps is not None:
            replacements['maxFps'] = str(max_fps)
        lines = [line for line in lines if line.split(':',1)[0] not in replacements]
        lines.extend(f'{key}:{value}' for key,value in replacements.items())
        file.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        yield
    finally:
        file.write_bytes(original)
        backup.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag', required=True)
    ap.add_argument('--ticks', type=int, default=300)
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--frame-width', type=int, default=320)
    ap.add_argument('--frame-height', type=int, default=180)
    ap.add_argument('--vsync', choices=('keep','on','off'), default='keep')
    ap.add_argument('--max-fps', type=int, default=None)
    ap.add_argument('--visible', action='store_true')
    ap.add_argument('--idle-only',action='store_true',
                    help='matched neutral ticks to isolate ray I/O from mining/pickup')
    ap.add_argument('--fixed-fixture',action='store_true',
                    help='use the same fixed anchor (-82,64,-9) as the camera-ray live probe')
    ap.add_argument('--fixture-xyz',type=int,nargs=3,metavar=('X','Y','Z'),
                    help='pin same lab terrain and height across independent launches')
    ap.add_argument('--ray-profile',choices=('coarse','fine'),default='coarse',
                    help='experimental 13x9 or 25x17 bounded first-hit ray profile')
    ap.add_argument('--surface-on-demand',action='store_true',
                    help='explicit fresh grouped sample only every sixth sync tick')
    ap.add_argument('--surface-objects',action='store_true',
                    help='opt-in grouped first-hit screen objects, bounded by ray count')
    ap.add_argument('--surface-grid',action='store_true',
                    help='opt-in 13x9 first-hit screen rays from patched MCio')
    ap.add_argument('--mining-feedback',action='store_true',
                    help='opt-in own vanilla break/crack stage, no Torch')
    ap.add_argument('--status-v1',action='store_true',
                    help='require gmaincraft-patched MCio status option every tick (no Torch)')
    args = ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',args.run_tag):
        ap.error('Invalid run tag')
    if not 145 <= args.ticks <= 2000 or not 1 <= args.repeats <= 5:
        ap.error('Invalid ticks/repeats')
    if not 96 <= min(args.frame_width,args.frame_height) and max(args.frame_width,args.frame_height) <= 1920:
        ap.error('Invalid frame width/height')
    if args.ray_profile=='fine' and not (args.surface_grid or args.surface_objects):
        ap.error('--ray-profile fine requires a surface sensor')
    if args.surface_on_demand and not args.surface_objects:
        ap.error('--surface-on-demand requires --surface-objects')
    if args.surface_grid and args.surface_objects:
        ap.error('For throughput comparisons choose exactly one sensor format')
    if args.fixed_fixture and args.fixture_xyz is not None:
        ap.error('Choose either --fixed-fixture or --fixture-xyz')
    if args.max_fps is not None and not 10 <= args.max_fps <= 260:
        ap.error('max fps outside Minecraft option range 10..260')
    path = ROOT/'logs'/f'mcio_throughput_{args.run_tag}.json'
    world_name = 'MDL_Throughput_'+args.run_tag
    save = INSTANCE/'saves'/world_name
    if path.exists() or save.exists():
        ap.error('Refusing to overwrite output/world; use fresh --run-tag')
    world.WorldManager(mcio_dir=MCIO).copy('storage','DemoWorld','DemoInstance',world_name,force=False)
    opts = types.RunOptions(mcio_dir=MCIO,instance_name='DemoInstance',world_name=world_name,
                            mcio_mode=types.MCioMode.SYNC,hide_window=not args.visible,
                            gpu_lib=None if args.visible else 'nvidia',
                            width=args.frame_width,height=args.frame_height,
                            env_extra={**({ENV_NAME:'1'} if args.status_v1 else {}),
                                       **({MINING_ENV:'1'} if args.mining_feedback else {}),
                                       **({SURFACE_GRID_ENV:'1'} if args.surface_grid else {}),
                                       **({SURFACE_OBJECTS_ENV:'1'} if args.surface_objects else {}),
                                       **({ON_DEMAND_ENV:'1'} if args.surface_on_demand else {}),
                                       **({'GMAINCRAFT_SURFACE_RAY_PROFILE':'fine'} if args.ray_profile=='fine' else {})})
    with temporary_options(args.vsync,args.max_fps,args.run_tag):
        launcher=instance.Launcher(opts)
        ctrl=None
        try:
            started=time.perf_counter()
            launcher.launch(wait=False)
            ctrl=controller.ControllerSync(action_port=opts.action_port,
                    observation_port=opts.observation_port,connection_timeout=65)
            anchor=(-82,64,-9) if args.fixed_fixture else (tuple(args.fixture_xyz) if args.fixture_xyz is not None else None)
            xyz=setup(ctrl,anchor)
            print('READY seconds',round(time.perf_counter()-started,3),
                  'position',xyz,'vsync',args.vsync,'max_fps',args.max_fps,flush=True)
            results=[do_run(ctrl,xyz,args.ticks,f'repeat{i+1}',status_v1=args.status_v1,
                            surface_grid=args.surface_grid,surface_objects=args.surface_objects,
                            on_demand=args.surface_on_demand,idle_only=args.idle_only,
                            mining_feedback=args.mining_feedback)
                     for i in range(args.repeats)]
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x',encoding='utf-8') as f:
                json.dump(dict(config=dict(vars(args)),results=results),f,indent=2)
                f.write('\n')
        finally:
            if ctrl is not None:
                try: ctrl.send_stop()
                except Exception as e: print('Stop warning:',repr(e),flush=True)
                ctrl.close()
            launcher.close()
    print('SAVED',path,flush=True)


if __name__=='__main__':
    main()
