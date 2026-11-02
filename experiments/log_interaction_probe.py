"""Real Minecraft interaction probe: mouse look, mining and oak-log pickup.

This is an INSTRUMENTATION TEST, not learned intelligence. A privileged controller
places a trunk and teleports the player once, in a disposable copied world.
During the scored phase the client may use ONLY vanilla MCio input events:
mouse motion, left-click and W. No /give, /setblock or /tp after reset.
Success requires a log to appear in observed inventory after real mining.
Run in isolated Gentoo Shell with tools/run-mcio-viewer.sh ... --visible.
"""
import argparse
import math
import time
import sys
from pathlib import Path

from mcio_ctrl import controller, instance, network, types, world
from mcio_ctrl.envs import env_util, mcio_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gmaincraft.log_curriculum import validate_tag, inventory_count, relative_cursor
MCIO_DIR = ROOT / 'mcio-data'



class Client:
    def __init__(self, ctrl):
        self.ctrl = ctrl
        self.manager = env_util.InputStateManager()
        self.last = None
        self.frames = 0
        self.score_commands = []
        self.scoring = False

    def step(self, *, keys=(), look=None, commands=(), clear=False):
        if self.scoring and commands:
            raise AssertionError('Admin commands prohibited during mining/collection')
        if commands:
            self.score_commands.extend(commands)
        pressed = {name: int(name in keys) for name in ('W','LEFT_BUTTON')}
        inputs = self.manager.process_action(pressed, mcio_env.INPUT_MAP)
        pkt = network.ActionPacket(inputs=inputs, commands=list(commands), clear_input=clear)
        if look is not None:
            if self.last is None:
                raise ValueError('Initial observation missing')
            pkt.cursor_pos = relative_cursor(self.last, *look)
        self.ctrl.send_action(pkt)
        ob = self.ctrl.recv_observation()
        if ob is None or ob.version != network.MCIO_PROTOCOL_VERSION:
            raise RuntimeError('No valid Minecraft observation')
        self.last = ob
        self.frames += 1
        return ob

    def advance(self, ticks, keys=()):
        for _ in range(ticks):
            self.step(keys=keys)
        return self.last


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag', required=True, help='new isolated world name, refuses duplicates')
    ap.add_argument('--visible', action='store_true')
    ap.add_argument('--max-mining-ticks', type=int, default=100)
    args = ap.parse_args()
    if not 5 <= args.max_mining_ticks <= 240:
        ap.error('max-mining-ticks must be 5..240')
    try:
        world_name = validate_tag(args.run_tag)
    except ValueError as exc:
        ap.error(str(exc))
    save = MCIO_DIR / 'instances' / 'DemoInstance' / 'saves' / world_name
    if save.exists():
        ap.error(f'Disposable world already exists; choose another tag: {save}')

    print(f'Copying template into ONLY {world_name}', flush=True)
    world.WorldManager(mcio_dir=MCIO_DIR).copy(
        'storage','DemoWorld','DemoInstance',world_name,force=False)
    opts = types.RunOptions(mcio_dir=MCIO_DIR, instance_name='DemoInstance',
        world_name=world_name, mcio_mode=types.MCioMode.SYNC,
        hide_window=not args.visible, gpu_lib=None if args.visible else 'nvidia',
        width=640 if args.visible else 320, height=360 if args.visible else 180)
    launcher = instance.Launcher(opts)
    ctrl = None
    try:
        started = time.perf_counter()
        launcher.launch(wait=False)
        ctrl = controller.ControllerSync(action_port=opts.action_port,
            observation_port=opts.observation_port, connection_timeout=90)
        c = Client(ctrl)
        ob = c.step(clear=True)
        fx = math.floor(ob.player_pos[0]); fz=math.floor(ob.player_pos[2]);
        fy = math.floor(ob.player_pos[1]) - 1
        x, z = fx, fz+2
        commands = [
            'difficulty peaceful','gamerule doMobSpawning false',
            'gamerule doDaylightCycle false','time set day','weather clear',
            f'fill {fx-8} {fy+1} {fz-8} {fx+8} {fy+7} {fz+8} minecraft:air replace',
            f'fill {fx-8} {fy} {fz-8} {fx+8} {fy} {fz+8} minecraft:bedrock replace',
            f'setblock {x} {fy+1} {z} minecraft:oak_log replace',
            f'setblock {x} {fy+2} {z} minecraft:oak_log replace',
            f'setblock {x} {fy+3} {z} minecraft:oak_log replace',
            f'tp @s {fx+0.5:.3f} {fy+1.0:.3f} {fz+0.5:.3f} 0 0',
            'gamemode survival @s','clear @s','effect clear @s',
            'effect give @s minecraft:instant_health 1 5 true',
            'effect give @s minecraft:saturation 1 20 true',
        ]
        c.step(commands=commands,clear=True)
        c.advance(24)
        ob=c.last
        if inventory_count(ob):
            raise RuntimeError('Non-empty oak-log inventory after /clear')
        print(f'PREPARED pos={ob.player_pos} yaw={ob.player_yaw:.2f} pitch={ob.player_pitch:.2f} '
              f'pixels={ob.frame_width}x{ob.frame_height} bytes={len(ob.frame)} '
              f'cursor={ob.cursor_pos}',flush=True)
        print('Starting action-only phase; no administrative commands permitted.',flush=True)
        c.scoring=True

        # Calibrate real camera path with *mouse input* before the scored mining.
        before_yaw, before_pitch=ob.player_yaw,ob.player_pitch
        ob=c.step(look=(120,50))
        ob=c.advance(4)
        dyaw=(ob.player_yaw-before_yaw+180)%360-180
        dpitch=ob.player_pitch-before_pitch
        print(f'CAMERA mouse(+120,+50) -> delta_yaw={dyaw:.3f} '
              f'delta_pitch={dpitch:.3f} (expected approx +18,+7.5)',flush=True)
        if abs(dyaw)<1 or abs(dpitch)<0.5:
            raise RuntimeError('MCio mouse motion did not change the game camera')
        # Undo *with mouse* (never /tp in the action-only phase).
        ob=c.step(look=(-120,-50))
        ob=c.advance(4)
        print(f'CAMERA restored yaw={ob.player_yaw:.2f} pitch={ob.player_pitch:.2f}',flush=True)
        if abs((ob.player_yaw+180)%360-180)>5 or abs(ob.player_pitch)>5:
            raise RuntimeError('Could not restore crosshair to target')

        # Mine by holding left mouse click, normal survival vanilla mechanics.
        hit=False
        first_item=None
        for tick in range(1,args.max_mining_ticks+1):
            ob=c.step(keys=('LEFT_BUTTON',))
            count=inventory_count(ob)
            if count:
                first_item=tick
                hit=True
                break
            if tick in (1,20,40,60,80,100,args.max_mining_ticks):
                print(f'MINING tick={tick} inventory_oak={count} '
                      f'look={ob.player_yaw:.1f},{ob.player_pitch:.1f}',flush=True)
        # Stop attack; walk towards the dropped item on floor.
        ob=c.step()
        for tick in range(1,60):
            ob=c.step(keys=('W',) if tick<=25 else ())
            if inventory_count(ob):
                first_item = first_item or args.max_mining_ticks+tick
                hit=True
                print(f'PICKUP observed after {tick} approach ticks at {ob.player_pos}',flush=True)
                break
        print(f'RESULT acquired={hit} oak_count={inventory_count(ob)} '
              f'frame_seq={ob.frame_sequence} exchanges={c.frames} '
              f'elapsed={time.perf_counter()-started:.2f}s',flush=True)
        if not hit:
            raise RuntimeError('No oak log collected via real Minecraft actions')
        assert inventory_count(ob)>=1
        assert all(not cmd.startswith('give ') for cmd in commands)
        print('PASS: camera mouse input, vanilla mining and inventory-based oak-log acquisition',flush=True)
        print('LIMIT: this was a scripted input-path probe with a controller-placed log, not learned aiming or visual recognition.',flush=True)
    finally:
        if ctrl is not None:
            try: ctrl.send_stop()
            except Exception as exc: print('Stop warning:',repr(exc),flush=True)
            ctrl.close()
        launcher.close()


if __name__ == '__main__':
    main()
