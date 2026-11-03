"""Real Minecraft synchronized request-on-last-tick bridge protocol audit.

The experimental grouped scene memory must be freshly requested for a
particular normal action. Absence of a request must never replay old world
positions. This is lab validation, not a scripted speedrunning policy.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

from mcio_ctrl import controller,instance,types,world
from mcio_ctrl.envs import env_util

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.agent_interface import AgentAction,MCioAgentBridge
from gmaincraft.instance_options import temporary_game_options
from gmaincraft.mcio_status import ENV_NAME as STATUS_ENV
from gmaincraft.mcio_surface_objects import (ENV_NAME as OBJECTS_ENV,ON_DEMAND_ENV,
                                             SurfaceObjectsOption)
from gmaincraft.vision_settings import ensure_clean_visual_options
from mcio_throughput_probe import action,setup,reset

MCIO=ROOT/'mcio-data'
INSTANCE=MCIO/'instances'/'DemoInstance'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--visible',action='store_true')
    args=ap.parse_args()
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,31}',args.run_tag):
        ap.error('Invalid unique tag')
    out=ROOT/'logs'/f'mcio_surface_request_{args.run_tag}.json'
    world_name='MDL_SurfaceRequest_'+args.run_tag
    if out.exists() or (INSTANCE/'saves'/world_name).exists():
        ap.error('Refusing to overwrite old world or results')
    world.WorldManager(mcio_dir=MCIO).copy('storage','DemoWorld','DemoInstance',world_name,force=False)
    ensure_clean_visual_options(INSTANCE)
    options=types.RunOptions(mcio_dir=MCIO,instance_name='DemoInstance',world_name=world_name,
                 mcio_mode=types.MCioMode.SYNC,hide_window=not args.visible,
                 gpu_lib=None if args.visible else 'nvidia',width=320,height=180,
                 env_extra={STATUS_ENV:'1',OBJECTS_ENV:'1',ON_DEMAND_ENV:'1'})
    profile={'renderDistance':'4','enableVsync':'false','maxFps':'260'}
    with temporary_game_options(INSTANCE,profile,run_tag=args.run_tag):
        launcher=instance.Launcher(options)
        ctrl=None
        try:
            start=time.perf_counter()
            launcher.launch(wait=False)
            ctrl=controller.ControllerSync(action_port=options.action_port,
                    observation_port=options.observation_port,connection_timeout=65)
            xyz=setup(ctrl)
            reset(ctrl,xyz)
            initial,_,_=action(ctrl,env_util.InputStateManager())
            bridge=MCioAgentBridge(ctrl,initial)
            checks={'not_sent_without_request':initial.get_option(SurfaceObjectsOption) is None}
            print('REQUEST READY',round(time.perf_counter()-start,3),xyz,flush=True)

            start_io=time.perf_counter()
            bridge.apply(AgentAction(ticks=6),request_surface_objects=True)
            first=bridge.surface_objects()
            elapsed=time.perf_counter()-start_io
            checks['fresh_oak_when_requested']=bool(first.matching('minecraft:oak_log'))
            checks['synced_last_tick_request']=first.schema=='gmaincraft_surface_objects_v1'
            first_seq=bridge.packet.last_action_sequence
            print('FRESH',len(first.objects),'oak',len(first.matching('minecraft:oak_log')),
                  'elapsed',round(elapsed,4),flush=True)

            bridge.apply(AgentAction(ticks=3))
            checks['no_stale_packet_on_non_request']=bridge.packet.get_option(SurfaceObjectsOption) is None
            try:
                bridge.surface_objects()
            except RuntimeError:
                checks['adapter_fails_closed_on_missing_sample']=True
            else:
                checks['adapter_fails_closed_on_missing_sample']=False
            bridge.apply(AgentAction(look_dx=850,ticks=6),request_surface_objects=True)
            away=bridge.surface_objects()
            checks['new_camera_not_old_oak']=not away.matching('minecraft:oak_log')
            bridge.apply(AgentAction(ticks=1),request_surface_objects=True)
            instant=bridge.surface_objects()
            checks['single_tick_fresh_request']=bool(instant.objects)
            checks['valid_sequential_packets']=bridge.packet.last_action_sequence>first_seq
            report=dict(tag=args.run_tag,fixture=xyz,profile=profile,checks=checks,
                        first=dict(objects=len(first.objects),oak=len(first.matching('minecraft:oak_log')),
                                   game_io_s=round(elapsed,4)),
                        turned_away=dict(objects=len(away.objects),
                                         oak=len(away.matching('minecraft:oak_log'))),
                        instant=dict(objects=len(instant.objects)),
                        final_packet_seq=bridge.packet.last_action_sequence)
            out.parent.mkdir(parents=True,exist_ok=True)
            with out.open('x',encoding='utf8') as stream:
                json.dump(report,stream,indent=2)
                stream.write('\n')
            print('REQUEST AUDIT',checks,'all_passed',all(checks.values()),flush=True)
            print('SAVED',out,flush=True)
            if not all(checks.values()):
                raise AssertionError('Request sampling did not match real game action/observation')
        finally:
            if ctrl is not None:
                try:ctrl.send_stop()
                except Exception as exc:print('STOP WARNING',repr(exc),flush=True)
                ctrl.close()
            launcher.close()

if __name__=='__main__':main()
