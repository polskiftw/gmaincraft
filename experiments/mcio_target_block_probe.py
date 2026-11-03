"""Live opt-in Fabric MCio player-status and VANILLA crosshair-hit audit.

One fresh disposable world. A tree and later an opaque occluder are staged by
controller commands OUTSIDE scored sensor phases. The observed target is read
from the real Minecraft client raycast, not from world/chunk scans. Turning
uses only vanilla MCio mouse movement. No Torch or human/external model.

Requires the gmaincraft MCio patch installed in isolated DemoInstance.
"""
import argparse
from dataclasses import asdict
import json
import math
import re
import sys
import time
from pathlib import Path

from mcio_ctrl import controller, instance, types, world
from mcio_ctrl.envs import env_util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from gmaincraft.agent_interface import AgentAction, MCioAgentBridge
from gmaincraft.instance_options import temporary_game_options
# IMPORTANT: registers new type with MCio CBOR BEFORE receiving observations.
from gmaincraft.mcio_status import ENV_NAME, PlayerStatusOption, PlayerStatusV1
from gmaincraft.vision_settings import ensure_clean_visual_options
from mcio_throughput_probe import action, setup, reset

MCIO=ROOT/'mcio-data'
INSTANCE=MCIO/'instances'/'DemoInstance'


def inspect(bridge, label, *, ticks=12):
    """Normal neutral synced frames; inspect only game reported hit result."""
    t0=time.perf_counter()
    bridge.apply(AgentAction(ticks=ticks))
    elapsed=time.perf_counter()-t0
    status=bridge.player_status()
    player=bridge.player_knowledge()
    item=status.crosshair_block
    result=dict(label=label,ticks=ticks,game_io_wall_s=round(elapsed,4),
                game_io_ms_per_tick=round(1000*elapsed/ticks,3),
                player_pos=tuple(round(x,3) for x in player.player_xyz),
                player_yaw=round(player.yaw_degrees,3),
                player_pitch=round(player.pitch_degrees,3),
                status=asdict(status),frame=[bridge.packet.frame_width,bridge.packet.frame_height],
                packet_seq=bridge.packet.last_action_sequence)
    print('SENSOR',json.dumps(result),flush=True)
    return result


def calibrate_vanilla_crosshair(ctrl, xyz):
    """Empirically query vanilla survivor crosshair across known lab fixtures.

    Controller may stage blocks/pose OUTSIDE any scored agent gameplay. The
    playing agent would see ONLY the actual Minecraft hit, not fixture data.
    Distinct from estimating general visual visibility/FOV of distant blocks.
    """
    x,y,z=xyz
    manager=env_util.InputStateManager()
    results=[]
    for offset in (2,3,4,5,6,7,8):
        commands=[
            f'fill {x-1} {y+1} {z+1} {x+1} {y+4} {z+9} minecraft:air replace minecraft:oak_log',
            f'fill {x-1} {y+1} {z+1} {x+1} {y+4} {z+9} minecraft:air replace minecraft:stone',
            f'fill {x-1} {y+1} {z+1} {x+1} {y+4} {z+9} minecraft:air replace minecraft:glass',
            f'setblock {x} {y+2} {z+offset} minecraft:oak_log replace',
            f'tp @s {x+.5} {y+1.0} {z+.5} 0 0',
        ]
        obs,_,_=action(ctrl,manager,commands=commands,clear=True)
        for _ in range(9):obs,_,_=action(ctrl,manager)
        status=PlayerStatusV1.from_packet(obs)
        if abs(obs.player_yaw)>1 or abs(obs.player_pitch)>1:
            raise AssertionError('Camera reset did not settle during reach calibration')
        target=status.crosshair_block
        row=dict(block_offset=offset,expected_fixture_id='minecraft:oak_log',
                 hit_id=(target.block_id if target else None),
                 hit_distance=(round(target.distance,4) if target else None),
                 hit_xyz=(target.xyz if target else None))
        results.append(row)
        print('REACH',json.dumps(row),flush=True)
    # Visible through glass does not mean interactable through glass: Minecraft
    # crosshair targets the intervening transparent solid block instead.
    commands=[
        f'fill {x-1} {y+1} {z+1} {x+1} {y+4} {z+9} minecraft:air replace minecraft:oak_log',
        f'setblock {x} {y+2} {z+2} minecraft:oak_log replace',
        f'setblock {x} {y+2} {z+1} minecraft:glass replace',
        f'tp @s {x+.5} {y+1.0} {z+.5} 0 0',
    ]
    obs,_,_=action(ctrl,manager,commands=commands,clear=True)
    for _ in range(10):obs,_,_=action(ctrl,manager)
    glass=PlayerStatusV1.from_packet(obs).crosshair_block
    through_glass=dict(expected='minecraft:glass',
                       hit_id=(glass.block_id if glass else None),
                       hit_distance=(round(glass.distance,4) if glass else None))
    print('TRANSPARENCY',json.dumps(through_glass),flush=True)
    # Scientifically useful bound: exact vanilla crosshair reach is at most
    # the last offset showing a real oak hit, but this is NOT a visibility
    # radius for anything else (nearby offset vs front-face distance differs).
    detected=[r for r in results if r['hit_id']=='minecraft:oak_log']
    return dict(reach_trials=results,glass_occlusion=through_glass,
                furthest_observed_offset=max((r['block_offset'] for r in detected),default=None),
                near_detected=results[0]['hit_id']=='minecraft:oak_log',
                far_suppressed=all(r['hit_id']!='minecraft:oak_log' for r in results[-2:]),
                glass_blocks_hit=through_glass['hit_id']=='minecraft:glass')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--visible',action='store_true')
    ap.add_argument('--calibrate',action='store_true',
                    help='also stage varied-distance and transparent block fixtures')
    ap.add_argument('--render-distance',type=int,default=4)
    ap.add_argument('--frame-width',type=int,default=320)
    ap.add_argument('--frame-height',type=int,default=180)
    args=ap.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',args.run_tag):
        ap.error('Invalid run tag')
    if not 2<=args.render_distance<=32:
        ap.error('Render distance 2..32')
    out=ROOT/'logs'/f'mcio_target_{args.run_tag}.json'
    world_name='MDL_TargetSensor_'+args.run_tag
    save=INSTANCE/'saves'/world_name
    if save.exists() or out.exists():
        ap.error('Refusing to overwrite previous world or output')
    world.WorldManager(mcio_dir=MCIO).copy('storage','DemoWorld','DemoInstance',world_name,force=False)
    ensure_clean_visual_options(INSTANCE)
    config={'renderDistance':str(args.render_distance),'maxFps':'260','enableVsync':'false'}
    options=types.RunOptions(mcio_dir=MCIO,instance_name='DemoInstance',world_name=world_name,
                            mcio_mode=types.MCioMode.SYNC,hide_window=not args.visible,
                            gpu_lib=None if args.visible else 'nvidia',
                            width=args.frame_width,height=args.frame_height,
                            env_extra={ENV_NAME:'1'})
    results=[]
    with temporary_game_options(INSTANCE,config,run_tag=args.run_tag):
        launcher=instance.Launcher(options)
        ctrl=None
        try:
            startup=time.perf_counter()
            launcher.launch(wait=False)
            ctrl=controller.ControllerSync(action_port=options.action_port,
                    observation_port=options.observation_port,connection_timeout=65)
            # Verify new CBOR subtype right away. Test lives in patched game.
            xyz=setup(ctrl)
            reset(ctrl,xyz)
            initial,_,_=action(ctrl, env_util.InputStateManager())
            if initial.get_option(PlayerStatusOption) is None:
                raise RuntimeError('Patched mod did not send expected PlayerStatusOption')
            bridge=MCioAgentBridge(ctrl,initial)
            print('SENSOR READY',round(time.perf_counter()-startup,3), 'fixture',xyz,flush=True)
            first=inspect(bridge,'oak_visible',ticks=18)
            results.append(first)
            # Rotate the ordinary mouse camera sharply away, no teleport.
            bridge.apply(AgentAction(look_dx=850,ticks=3))
            away=inspect(bridge,'turned_away',ticks=12)
            results.append(away)
            # Controller-only fixture reset, not a scored agent action.
            reset(ctrl,xyz)
            after_reset,_,_=action(ctrl,env_util.InputStateManager())
            bridge=MCioAgentBridge(ctrl,after_reset)
            again=inspect(bridge,'oak_restored',ticks=12)
            results.append(again)
            # Insert stone between player and oak (forbidden to the playing
            # agent: controller-only stage command), then resample crosshair.
            x,y,z=xyz
            manager=env_util.InputStateManager()
            for command in (f'setblock {x} {y+1} {z+1} minecraft:stone replace',
                            f'setblock {x} {y+2} {z+1} minecraft:stone replace'):
                obs,_,_=action(ctrl,manager,commands=(command,))
            for _ in range(4): obs,_,_=action(ctrl,manager)
            bridge=MCioAgentBridge(ctrl,obs)
            blocked=inspect(bridge,'stone_occludes_oak',ticks=12)
            results.append(blocked)
            # Positive+negative controls must use client-derived targets.
            oak=lambda item: item['status']['crosshair_block'] and item['status']['crosshair_block']['block_id']=='minecraft:oak_log'
            stone=lambda item: item['status']['crosshair_block'] and item['status']['crosshair_block']['block_id']=='minecraft:stone'
            outcomes={'oak_visible':bool(oak(first)),
                      'turned_away_does_not_report_oak':not bool(oak(away)),
                      'oak_restored':bool(oak(again)),
                      'stone_occludes_oak':bool(stone(blocked)) and not bool(oak(blocked))}
            calibration=calibrate_vanilla_crosshair(ctrl,xyz) if args.calibrate else None
            if calibration is not None:
                outcomes['reach_near_hit']=calibration['near_detected']
                outcomes['reach_far_no_oracle']=calibration['far_suppressed']
                outcomes['glass_occlusion']=calibration['glass_blocks_hit']
            report=dict(tag=args.run_tag,world=world_name,options=config,
                        patched_mod_environment=ENV_NAME,checks=outcomes,results=results,
                        calibration=calibration)
            out.parent.mkdir(parents=True,exist_ok=True)
            with out.open('x',encoding='utf8') as f:
                json.dump(report,f,indent=2)
                f.write('\n')
            print('SENSOR AUDIT',outcomes,'all_passed',all(outcomes.values()),flush=True)
            print('SAVED',out,flush=True)
            if not all(outcomes.values()):
                raise AssertionError('Sensor crosshair positive/negative scene control failed')
        finally:
            if ctrl is not None:
                try:ctrl.send_stop()
                except Exception as exc:print('STOP WARNING',repr(exc),flush=True)
                ctrl.close()
            launcher.close()

if __name__=='__main__':main()
