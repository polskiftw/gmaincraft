"""Real Minecraft paired-frame/first-hit object visibility falsification lab.

This experiment uses controller commands ONLY to prepare a disposable world.
It compares two ACTUAL MCio RGB captures (block absent/present) against an
on-demand read-only game-memory camera ray observation at a known position.
Screenshots are evaluator-only truth checks, NEVER policy inputs/rewards.
Visible pixel changes do not automatically imply human-identifiable blocks.
"""
import argparse
from dataclasses import asdict
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from mcio_ctrl import controller,instance,types,world
from mcio_ctrl.envs import env_util

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.instance_options import temporary_game_options
from gmaincraft.mcio_status import ENV_NAME as STATUS_ENV
from gmaincraft.mcio_surface_objects import (SurfaceObjectsV1,SurfaceObjectsOption,
                                             ON_DEMAND_ENV, ENV_NAME as OBJECTS_ENV)
from gmaincraft.surface_visibility_audit import compare_frames
from gmaincraft.vision_settings import ensure_clean_visual_options
from mcio_throughput_probe import action, setup, reset

MCIO=ROOT/'mcio-data'
INSTANCE=MCIO/'instances'/'DemoInstance'
TARGET='minecraft:magenta_concrete'
# One block in an otherwise unchanged scene, tested at varied projected sizes
# and horizontal positions. These are test inputs, NOT visibility thresholds.
GEOMETRIES=((0,3),(2,3),(4,3),(0,6),(1,6),(3,6),(6,6),
            (0,10),(1,10),(2,10),(4,10),(8,10),(0,16),(1,16),
            (3,16),(6,16),(0,24),(1,24),(3,24),(6,24),
            (0,32),(1,32),(3,32))


def command(ctrl, manager, instruction):
    # MCio can execute hundreds of render observations per real server tick.
    # A command is accepted by the client before its server block update
    # reaches the client. Explicit wall-clock settling avoids measuring the
    # PREVIOUS fixture as though it were the newly staged block.
    action(ctrl,manager,commands=(instruction,),clear=True)
    time.sleep(.15)
    for _ in range(12):
        action(ctrl,manager)


def capture(ctrl, manager, *, settle=4):
    for _ in range(settle-1): action(ctrl,manager)
    obs,_,_=action(ctrl,manager,request_objects=True)
    grouped=SurfaceObjectsV1.from_packet(obs)
    frame=np.array(obs.get_frame(),copy=True)
    if frame.shape!=(180,320,3):
        raise AssertionError(f'Unexpected real Minecraft RGB frame shape: {frame.shape}')
    return obs,grouped,frame


def record_scene(ctrl,manager,xyz,offset,distance,label,wall=None,night=False):
    x,y,z=xyz
    # One known lab target. Selection is by exact block XYZ; block types
    # behind known walls are evaluator labels only, not player knowledge.
    pos=(x+offset,y+2,z+distance)
    if wall is not None:
        barrier={'stone':'minecraft:stone','glass':'minecraft:glass',
                 'leaves':'minecraft:oak_leaves','slit':'minecraft:stone'}[wall]
        command(ctrl,manager,f'fill {x-8} {y+1} {z+4} {x+8} {y+7} {z+4} {barrier} replace')
        if wall=='slit':
            command(ctrl,manager,f'fill {x} {y+2} {z+4} {x} {y+3} {z+4} minecraft:air replace')
    if night:
        command(ctrl,manager,'time set midnight')
    command(ctrl,manager,f'setblock {pos[0]} {pos[1]} {pos[2]} minecraft:air replace')
    before,previous,old_rgb=capture(ctrl,manager)
    command(ctrl,manager,f'setblock {pos[0]} {pos[1]} {pos[2]} {TARGET} replace')
    after,grouped,new_rgb=capture(ctrl,manager)
    # Critical independent guard: the client can lag a server command by a
    # number of synchronized MCio ticks. Confirm the displayed frame and
    # target-sample membership are stable across ANOTHER real server tick.
    time.sleep(.11)
    confirmed,confirmed_grouped,confirmation_rgb=capture(ctrl,manager,settle=6)
    stability=compare_frames(new_rgb,confirmation_rgb,crosshair_radius=0)
    detected_on_confirmation=any(obj.xyz==pos and obj.block_id==TARGET
                                 for obj in confirmed_grouped.objects)
    change=compare_frames(old_rgb,new_rgb)
    full_change=compare_frames(old_rgb,new_rgb,crosshair_radius=0)
    detected=tuple(obj for obj in grouped.objects if obj.xyz==pos and obj.block_id==TARGET)
    was_present=any(obj.xyz==pos and obj.block_id==TARGET for obj in previous.objects)
    result=dict(label=label,position=pos,lateral_offset=offset,distance_blocks=distance,
                pose_before={'pos':before.player_pos,'yaw':before.player_yaw,'pitch':before.player_pitch},
                pose_after={'pos':after.player_pos,'yaw':after.player_yaw,'pitch':after.player_pitch},
                nearest_objects=[dict(xyz=o.xyz,block=o.block_id,distance=round(o.nearest_distance,2),
                        uv=o.screen_center_uv) for o in sorted(grouped.objects,key=lambda o:o.nearest_distance)[:8]],
                wall=wall,night=night,frame_sequence=[before.frame_sequence,after.frame_sequence,confirmed.frame_sequence],
                confirmation_changed_pixels=stability.changed_pixels,
                confirmed_detected=detected_on_confirmation,
                pixels=asdict(change),pixels_without_crosshair_mask=asdict(full_change),
                sample_grid=[grouped.columns,grouped.rows],candidate_count=len(grouped.objects),
                target_detected=bool(detected),target_ray_samples=sum(len(v.ray_indices) for v in detected),
                target_projected_uv=[v.screen_center_uv for v in detected],
                target_light=[(v.sky_light,v.block_light) for v in detected],
                unexpectedly_in_before=was_present)
    print('VISIBILITY',json.dumps(result,sort_keys=True),flush=True)
    command(ctrl,manager,f'setblock {pos[0]} {pos[1]} {pos[2]} minecraft:air replace')
    if night:
        command(ctrl,manager,'time set noon')
    if wall is not None:
        command(ctrl,manager,f'fill {x-8} {y+1} {z+4} {x+8} {y+7} {z+4} minecraft:air replace')
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--suite',choices=('geometry','occlusion','diagnose'),required=True)
    ap.add_argument('--visible',action='store_true')
    ap.add_argument('--ray-profile',choices=('coarse','fine'),default='coarse')
    args=ap.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',args.run_tag):
        ap.error('Invalid run tag')
    output=ROOT/'logs'/f'mcio_visibility_{args.run_tag}.json'
    world_name='MDL_VisCal_'+args.run_tag
    if output.exists() or (INSTANCE/'saves'/world_name).exists():
        ap.error('Refusing to overwrite old experiment or game world')
    world.WorldManager(mcio_dir=MCIO).copy('storage','DemoWorld','DemoInstance',world_name,force=False)
    ensure_clean_visual_options(INSTANCE)
    profile={'renderDistance':'4','enableVsync':'false','maxFps':'260'}
    opts=types.RunOptions(mcio_dir=MCIO,instance_name='DemoInstance',world_name=world_name,
         mcio_mode=types.MCioMode.SYNC,hide_window=not args.visible,
         gpu_lib=None if args.visible else 'nvidia',width=320,height=180,
         env_extra={STATUS_ENV:'1',OBJECTS_ENV:'1',ON_DEMAND_ENV:'1',
                    **({'GMAINCRAFT_SURFACE_RAY_PROFILE':'fine'} if args.ray_profile=='fine' else {})})
    with temporary_game_options(INSTANCE,profile,run_tag=args.run_tag):
        launcher=instance.Launcher(opts)
        ctrl=None
        try:
            began=time.perf_counter()
            launcher.launch(wait=False)
            ctrl=controller.ControllerSync(action_port=opts.action_port,
                        observation_port=opts.observation_port,connection_timeout=65)
            xyz=setup(ctrl)
            reset(ctrl,xyz)
            manager=env_util.InputStateManager()
            x,y,z=xyz
            # Dedicated unchanged camera and corridor for paired captures.
            # Full-game admin operations are lab-only, not policy gameplay.
            command(ctrl,manager,f'fill {x-8} {y+1} {z+1} {x+8} {y+8} {z+35} minecraft:air replace')
            command(ctrl,manager,f'fill {x-8} {y} {z+1} {x+8} {y} {z+35} minecraft:bedrock replace')
            command(ctrl,manager,f'tp @s {x+.5} {y+1} {z+.5} 0 0')
            command(ctrl,manager,'time set noon')
            obs0,group0,img0=capture(ctrl,manager,settle=42)
            obs1,group1,img1=capture(ctrl,manager,settle=6)
            control=compare_frames(img0,img1)
            print('VISIBILITY READY',round(time.perf_counter()-began,3),xyz,
                  'pose',obs0.player_pos,obs0.player_yaw,obs0.player_pitch,
                  'nearest',[(o.xyz,o.block_id,round(o.nearest_distance,2)) for o in sorted(group0.objects,key=lambda o:o.nearest_distance)[:8]],
                  'image_control',asdict(control),flush=True)
            results=[]
            if args.suite=='diagnose':
                for dx,dz in ((0,3),(2,3),(0,6),(3,6)):
                    results.append(record_scene(ctrl,manager,xyz,dx,dz,f'diagnostic_{dx}_{dz}'))
            elif args.suite=='geometry':
                for dx,dz in GEOMETRIES:
                    results.append(record_scene(ctrl,manager,xyz,dx,dz,f'offset{dx}_depth{dz}'))
            else:
                for material in ('stone','glass','leaves','slit'):
                    results.append(record_scene(ctrl,manager,xyz,0,10,material,wall=material))
                results.append(record_scene(ctrl,manager,xyz,0,10,'night',night=True))
                results.append(record_scene(ctrl,manager,xyz,3,24,'far_night',night=True))
            same_pose=all(row['frame_sequence'][0]<row['frame_sequence'][1]<row['frame_sequence'][2]
                          for row in results)
            no_stale=all(not row['unexpectedly_in_before'] for row in results)
            scene_stable=all(row['confirmation_changed_pixels']==0 and
                             row['confirmed_detected']==row['target_detected']
                             for row in results)
            outcome=dict(frame_sequences_advance=same_pose,no_fixture_stale=no_stale,
                         unchanged_frame_stable=control.changed_pixels==0,
                         every_final_scene_stable=scene_stable)
            report=dict(tag=args.run_tag,suite=args.suite,profile=profile,ray_profile=args.ray_profile,world=world_name,
                        fixture=xyz,model_input=False,target_block=TARGET,
                        rgb_change_threshold=40,results=results,checks=outcome,
                        unchanged_frame_control=asdict(control),total_elapsed_s=round(time.perf_counter()-began,3))
            output.parent.mkdir(parents=True,exist_ok=True)
            with output.open('x',encoding='utf8') as stream:
                json.dump(report,stream,indent=2)
                stream.write('\n')
            print('VISIBILITY AUDIT',json.dumps(outcome),'all_passed',all(outcome.values()),flush=True)
            print('SAVED',output,flush=True)
            if not all(outcome.values()):
                raise AssertionError('Paired-frame evidence validity check failed')
        finally:
            if ctrl is not None:
                try:ctrl.send_stop()
                except Exception as exc:print('STOP WARNING',repr(exc),flush=True)
                ctrl.close()
            launcher.close()

if __name__=='__main__':main()
