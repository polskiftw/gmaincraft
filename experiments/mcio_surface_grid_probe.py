"""Live experimental camera-ray vs actual Minecraft visibility lab fixtures.

No learned model, no Torch. Commands ONLY prepare disposable, non-scored
controlled worlds. The sensor itself is read-only, bounded first-hit rays;
this probe records failures and cannot make the research sensor policy-eligible.
One Minecraft launch per unique tag. Use HEADLESS Gamescope for throughput.
"""
import argparse
from dataclasses import asdict
import json
import re
import sys
import time
from pathlib import Path

from mcio_ctrl import controller, instance, types, world
from mcio_ctrl.envs import env_util

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.instance_options import temporary_game_options
from gmaincraft.mcio_status import PlayerStatusV1, ENV_NAME as STATUS_ENV
from gmaincraft.mcio_surface_grid import SurfaceGridV0, ENV_NAME as GRID_ENV
from gmaincraft.mcio_surface_objects import SurfaceObjectsV1, ENV_NAME as OBJECTS_ENV
from gmaincraft.vision_settings import ensure_clean_visual_options
from mcio_throughput_probe import action, setup, reset

MCIO=ROOT/'mcio-data'
INSTANCE=MCIO/'instances'/'DemoInstance'


def observe(ctrl, manager, label, settle=12, grouped=False):
    start=time.perf_counter()
    obs=None
    for _ in range(settle):
        obs,_,_=action(ctrl,manager)
    grid=SurfaceGridV0.from_packet(obs)
    status=PlayerStatusV1.from_packet(obs)
    center=(grid.rows//2)*grid.columns+(grid.columns//2)
    center_hit=next((hit for hit in grid.hits if hit.index==center),None)
    oak=grid.block_hits('minecraft:oak_log')
    objects=SurfaceObjectsV1.from_packet(obs) if grouped else None
    if objects is not None:
        objects.assert_same_first_hits(grid)
    results=dict(label=label,game_io_s=round(time.perf_counter()-start,4),
                 ticks=settle,grid=dict(schema=grid.schema,shape=[grid.columns,grid.rows],
                 fov=grid.vertical_fov_degrees,max_distance=grid.max_distance,
                 total_hits=len(grid.hits),oak_rays=len(oak),
                 unique_oak_positions=sorted({hit.xyz for hit in oak}),
                 oak_screen_uv=[hit.normalized_uv for hit in oak],
                 center=asdict(center_hit) if center_hit else None,
                 sample_hits=[asdict(hit) for hit in grid.hits]),
                 crosshair=asdict(status.crosshair_block) if status.crosshair_block else None,
                 grouped=dict(schema=objects.schema,distinct_blocks=len(objects.objects),
                       oak_objects=len(objects.matching('minecraft:oak_log')),
                       ray_samples=sum(len(obj.ray_indices) for obj in objects.objects),
                       objects=[asdict(item) for item in objects.objects]) if objects else None,
                 action_seq=obs.last_action_sequence,frame_seq=obs.frame_sequence)
    print('SURFACE',json.dumps({k:v for k,v in results.items() if k!='grid'}|
             {'grid':{k:v for k,v in results['grid'].items() if k!='sample_hits'},
              'grouped':{k:v for k,v in (results['grouped'] or {}).items() if k!='objects'}},
             sort_keys=True),flush=True)
    return results


def fixture(ctrl,manager,xyz,commands):
    x,y,z=xyz
    # Restore a bounded lab volume only; wall and oak changes never persist
    # outside THIS copied disposable fixture world. Avoid chunk-scanning.
    base=[f'fill {x-4} {y+1} {z+1} {x+4} {y+7} {z+10} minecraft:air replace minecraft:oak_log',
          f'fill {x-4} {y+1} {z+1} {x+4} {y+7} {z+10} minecraft:air replace minecraft:stone',
          f'fill {x-4} {y+1} {z+1} {x+4} {y+7} {z+10} minecraft:air replace minecraft:glass',
          f'tp @s {x+.5} {y+1} {z+.5} 0 0']
    for command in base+commands:
        action(ctrl,manager,commands=(command,),clear=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--visible',action='store_true')
    ap.add_argument('--grouped',action='store_true',help='audit first-hit grouped/ray parity')
    ap.add_argument('--render-distance',type=int,default=4)
    args=ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',args.run_tag):
        ap.error('Invalid tag')
    if not 2<=args.render_distance<=32:
        ap.error('Invalid render distance')
    output=ROOT/'logs'/f'mcio_surface_grid_{args.run_tag}.json'
    world_name='MDL_SurfaceGrid_'+args.run_tag
    if output.exists() or (INSTANCE/'saves'/world_name).exists():
        ap.error('Fresh run tag required: refusing overwrite')
    world.WorldManager(mcio_dir=MCIO).copy('storage','DemoWorld','DemoInstance',world_name,force=False)
    ensure_clean_visual_options(INSTANCE)
    config={'renderDistance':str(args.render_distance),'enableVsync':'false','maxFps':'260'}
    opts=types.RunOptions(mcio_dir=MCIO,instance_name='DemoInstance',world_name=world_name,
                          mcio_mode=types.MCioMode.SYNC,hide_window=not args.visible,
                          gpu_lib=None if args.visible else 'nvidia',width=320,height=180,
                          env_extra={STATUS_ENV:'1',GRID_ENV:'1',
                                    **({OBJECTS_ENV:'1'} if args.grouped else {})})
    with temporary_game_options(INSTANCE,config,run_tag=args.run_tag):
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
            print('SURFACE READY',round(time.perf_counter()-began,3),'fixture',xyz,flush=True)
            x,y,z=xyz
            runs=[]
            # Center oak is also the real vanilla crosshair-selected hit.
            fixture(ctrl,manager,xyz,[f'setblock {x} {y+2} {z+2} minecraft:oak_log replace'])
            runs.append(observe(ctrl,manager,'center_oak',grouped=args.grouped))
            # Oak now sits 2 blocks to the side, visible within the FOV but
            # not vanilla-selectable by the crosshair at this orientation.
            fixture(ctrl,manager,xyz,[f'setblock {x+2} {y+2} {z+4} minecraft:oak_log replace'])
            runs.append(observe(ctrl,manager,'off_center_oak',grouped=args.grouped))
            # Opaque stone fully intercepts line of sight to that side tree.
            fixture(ctrl,manager,xyz,[f'setblock {x+2} {y+2} {z+4} minecraft:oak_log replace',
                 f'fill {x-1} {y+1} {z+2} {x+3} {y+6} {z+2} minecraft:stone replace'])
            runs.append(observe(ctrl,manager,'stone_occlusion',grouped=args.grouped))
            # Player can see through transparent glass, but COLLIDER rays
            # cannot: deliberately measure this known omission, no X-ray.
            fixture(ctrl,manager,xyz,[f'setblock {x+2} {y+2} {z+4} minecraft:oak_log replace',
                 f'fill {x-1} {y+1} {z+2} {x+3} {y+6} {z+2} minecraft:glass replace'])
            runs.append(observe(ctrl,manager,'glass_occlusion',grouped=args.grouped))
            checks=dict(center_oak_hit=bool(runs[0]['grid']['oak_rays']),
                        off_center_oak_hit=bool(runs[1]['grid']['oak_rays']),
                        off_center_not_crosshair=(runs[1]['crosshair'] or {}).get('block_id')!='minecraft:oak_log',
                        stone_no_xray=not runs[2]['grid']['oak_rays'],
                        glass_conservative_no_xray=not runs[3]['grid']['oak_rays'])
            if args.grouped:
                checks['grouping_parity']=all(r['grouped']['ray_samples']==r['grid']['total_hits'] for r in runs)
            report=dict(tag=args.run_tag,options=config,fixture=xyz,
                        outcomes=checks,results=runs)
            output.parent.mkdir(parents=True,exist_ok=True)
            with output.open('x',encoding='utf8') as stream:
                json.dump(report,stream,indent=2)
                stream.write('\n')
            print('SURFACE AUDIT',checks,'all_passed',all(checks.values()),flush=True)
            print('SAVED',output,flush=True)
            if not all(checks.values()):
                raise AssertionError('Live surface ray controls disagree with expected observations')
        finally:
            if ctrl is not None:
                try:ctrl.send_stop()
                except Exception as exc: print('STOP WARNING',repr(exc),flush=True)
                ctrl.close()
            launcher.close()

if __name__=='__main__':main()
