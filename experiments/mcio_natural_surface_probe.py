"""No-Torch, live Minecraft natural-survival on-demand surface throughput.

Copies one already generated, verified noise world to a fresh disposable
instance save for EACH run tag. No game commands, scripted world edits, hidden
world-map observation or neural policy: only normal neutral game ticks and
optional first-collision camera ray requests. Compare identical game seeds,
spawns, options and 300-tick series across OFF/coarse/fine profiles.
"""
import argparse
from collections import Counter
import json
import re
import sys
import time
from pathlib import Path

from mcio_ctrl import controller,instance,types,world
from mcio_ctrl.envs import env_util
from nbt import nbt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.instance_options import temporary_game_options
from gmaincraft.mcio_surface_objects import (SurfaceObjectsOption,SurfaceObjectsV1,
                                              ENV_NAME as OBJECTS_ENV,ON_DEMAND_ENV)
from gmaincraft.vision_settings import ensure_clean_visual_options
from mcio_throughput_probe import action

MCIO=ROOT/'mcio-data'
INSTANCE=MCIO/'instances'/'DemoInstance'
DEFAULT_SOURCE='MDL_NaturalSource_statusnatural_1103a_01'


def source_metadata(source):
    data=nbt.NBTFile(str(source/'level.dat'))['Data']
    settings=data['WorldGenSettings']
    kind=settings['dimensions']['minecraft:overworld']['generator']['type'].value
    if kind!='minecraft:noise':
        raise RuntimeError('Refusing non-natural-world source: '+str(kind))
    return int(settings['seed'].value),kind


def measured(ctrl,*,profile,ticks,repeats):
    manager=env_util.InputStateManager()
    results=[]
    initial=None
    last_frame=None
    for rep in range(repeats):
        wall_start=time.perf_counter()
        starts=[]
        object_counts=[]
        by_id=Counter()
        sampled=0
        frame_gaps=0
        packet_gaps=0
        prev_seq=None
        check=0
        for i in range(ticks):
            request=profile!='off' and (i+1)%6==0
            obs,_,_=action(ctrl,manager,request_objects=request)
            if initial is None:
                initial=(tuple(round(float(v),3) for v in obs.player_pos),
                         float(obs.player_yaw),float(obs.player_pitch))
            if last_frame is not None and obs.frame_sequence!=last_frame+1:
                frame_gaps+=1
            last_frame=obs.frame_sequence
            if prev_seq is not None and obs.last_action_sequence!=prev_seq+1:
                packet_gaps+=1
            prev_seq=obs.last_action_sequence
            if request:
                view=SurfaceObjectsV1.from_packet(obs)
                assert (view.columns,view.rows)==((25,17) if profile=='fine' else (13,9))
                object_counts.append(len(view.objects))
                sampled+=1
                by_id.update({o.block_id for o in view.objects})
            elif obs.get_option(SurfaceObjectsOption) is not None:
                raise AssertionError('Unrequested packet leaked stale world objects')
            if i%6==5:
                img=obs.get_frame()
                if img.dtype.name!='uint8' or img.shape!=(180,320,3):
                    raise AssertionError('RGB game frame corrupted')
                check+=1
            if i%100==99:
                starts.append(round(time.perf_counter()-wall_start,4))
        elapsed=time.perf_counter()-wall_start
        result=dict(repeat=rep+1,ticks=ticks,wall_s=round(elapsed,4),
                    game_ticks_per_wall_s=round(ticks/elapsed,1),
                    cumulative_100_tick_times=starts,
                    surface_observations=sampled,
                    mean_distinct_blocks=round(sum(object_counts)/max(1,len(object_counts)),2),
                    top_sensed_block_ids=by_id.most_common(12),
                    frame_gaps=frame_gaps,packet_gaps=packet_gaps,
                    rgb_shapes_verified=check,
                    starting_camera=initial,final_camera=(tuple(round(float(v),3) for v in obs.player_pos),
                                               float(obs.player_yaw),float(obs.player_pitch)))
        results.append(result)
        print('NATURAL SENSOR',json.dumps(result,sort_keys=True),flush=True)
        if frame_gaps or packet_gaps or sampled!=(ticks//6 if profile!='off' else 0):
            raise AssertionError('Natural-scene synchronized sample/frame validation failed')
    return results


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--source-world',default=DEFAULT_SOURCE)
    ap.add_argument('--profile',choices=('off','coarse','fine'),required=True)
    ap.add_argument('--ticks',type=int,default=300)
    ap.add_argument('--repeats',type=int,default=2)
    ap.add_argument('--visible',action='store_true')
    args=ap.parse_args(argv)
    for value in (args.run_tag,args.source_world):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,64}',value):
            ap.error('Unsafe run tag/source world name')
    if args.ticks<6 or args.ticks>1200 or args.ticks%6 or not 1<=args.repeats<=3:
        ap.error('ticks must be a multiple of six, 6..1200, and repeats 1..3')
    source=MCIO/'world_storage'/args.source_world
    if not (source/'level.dat').is_file():
        ap.error('Saved natural source world missing: '+str(source))
    seed,kind=source_metadata(source)
    new_world='MDL_NaturalSensor_'+args.run_tag
    dest=INSTANCE/'saves'/new_world
    out=ROOT/'logs'/f'mcio_natural_surface_{args.run_tag}.json'
    if dest.exists() or out.exists():
        ap.error('Refusing to overwrite an existing world/report')
    world.WorldManager(mcio_dir=MCIO).copy('storage',args.source_world,'DemoInstance',new_world,force=False)
    ensure_clean_visual_options(INSTANCE)
    profile={'renderDistance':'4','enableVsync':'false','maxFps':'260'}
    env_extra=({OBJECTS_ENV:'1',ON_DEMAND_ENV:'1',
                **({'GMAINCRAFT_SURFACE_RAY_PROFILE':'fine'} if args.profile=='fine' else {})}
               if args.profile!='off' else {})
    opts=types.RunOptions(mcio_dir=MCIO,instance_name='DemoInstance',world_name=new_world,
         mcio_mode=types.MCioMode.SYNC,hide_window=not args.visible,
         gpu_lib=None if args.visible else 'nvidia',width=320,height=180,
         env_extra=env_extra)
    with temporary_game_options(INSTANCE,profile,run_tag=args.run_tag):
        launcher=instance.Launcher(opts)
        ctrl=None
        try:
            started=time.perf_counter()
            launcher.launch(wait=False)
            ctrl=controller.ControllerSync(action_port=opts.action_port,
                  observation_port=opts.observation_port,connection_timeout=65)
            print('NATURAL SENSOR READY',json.dumps(dict(tag=args.run_tag,source=args.source_world,
                  seed=seed,generator=kind,profile=args.profile,launch_s=round(time.perf_counter()-started,3))),flush=True)
            # First 300-tick repeat is cold/early world-rendering period,
            # second is warm. No admin commands are used during either run.
            results=measured(ctrl,profile=args.profile,ticks=args.ticks,repeats=args.repeats)
            report=dict(tag=args.run_tag,source=args.source_world,world=new_world,
                        seed=seed,generator=kind,profile=args.profile,
                        options=profile,ordinary_survival_actions_only=True,
                        results=results)
            out.parent.mkdir(parents=True,exist_ok=True)
            with out.open('x',encoding='utf8') as stream:
                json.dump(report,stream,indent=2)
                stream.write('\n')
            print('SAVED',out,flush=True)
        finally:
            if ctrl is not None:
                try:ctrl.send_stop()
                except Exception as exc:print('STOP WARNING',repr(exc),flush=True)
                ctrl.close()
            launcher.close()

if __name__=='__main__':main()
