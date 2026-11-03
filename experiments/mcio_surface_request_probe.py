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
from gmaincraft.mcio_mining_feedback import (MiningFeedbackOption,MiningFeedbackV1,
                                             ENV_NAME as MINING_ENV,
                                             ON_DEMAND_ENV as MINING_ON_DEMAND_ENV)
from gmaincraft.mcio_surface_objects import (ENV_NAME as OBJECTS_ENV,ON_DEMAND_ENV,
                                             SurfaceObjectsOption)
from gmaincraft.vision_settings import ensure_clean_visual_options
from mcio_throughput_probe import action,setup,reset

MCIO=ROOT/'mcio-data'
INSTANCE=MCIO/'instances'/'DemoInstance'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--with-mining',action='store_true',
                    help='verify simultaneous fresh surface and own mining packets')
    ap.add_argument('--visible',action='store_true')
    ap.add_argument('--vertical-extremes',action='store_true',
                    help='real-game ray camera-basis test at +/-90 degree pitch')
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
                 env_extra={STATUS_ENV:'1',OBJECTS_ENV:'1',ON_DEMAND_ENV:'1',
                            **({'GMAINCRAFT_SURFACE_RAY_PROFILE':'fine'} if args.vertical_extremes else {}),
                            **({MINING_ENV:'1',MINING_ON_DEMAND_ENV:'1'} if args.with_mining else {})})
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
            combined=None
            if args.with_mining:
                checks['mining_initially_absent']=initial.get_option(MiningFeedbackOption) is None
                bridge.apply(AgentAction(ticks=6),request_surface_objects=True,
                             request_mining_feedback=True)
                joint=bridge.surface_objects()
                feedback=bridge.mining_feedback()
                checks['combined_surface_and_mining_fresh']=bool(joint.objects) and feedback.schema=='gmaincraft_mining_feedback_v1'
                checks['combined_idle_no_attack']=not feedback.actively_destroying and feedback.crack_stage==-1
                bridge.apply(AgentAction(ticks=6))
                checks['both_options_absent_next_action']=(
                    bridge.packet.get_option(MiningFeedbackOption) is None and
                    bridge.packet.get_option(SurfaceObjectsOption) is None)
                try:
                    bridge.mining_feedback()
                except RuntimeError:
                    checks['missing_mining_fails_closed']=True
                else:
                    checks['missing_mining_fails_closed']=False
                combined=dict(objects=len(joint.objects),
                              crack_stage=feedback.crack_stage,
                              active=feedback.actively_destroying)
                print('JOINT SENSOR',json.dumps(combined),flush=True)
            vertical=None
            if args.vertical_extremes:
                # Stage off-center overhead oak OUTSIDE scored policy actions,
                # then use real mouse movement and the actual memory hit grid.
                reset(ctrl,xyz)
                manager=env_util.InputStateManager()
                x,y,z=xyz
                staged,_,_=action(ctrl,manager,commands=(
                    f'setblock {x+2} {y+7} {z} minecraft:oak_log replace',),clear=True)
                time.sleep(.2)  # server block update needs real wall-clock time
                for _ in range(20): staged,_,_=action(ctrl,manager)
                bridge=MCioAgentBridge(ctrl,staged)
                bridge.apply(AgentAction(look_dy=-1050,ticks=6),request_surface_objects=True)
                up_view=bridge.surface_objects()
                pitch_up=bridge.packet.player_pitch
                expected=(x+2,y+7,z)
                upper=next((o for o in up_view.objects if o.xyz==expected and
                            o.block_id=='minecraft:oak_log'),None)
                checks['vertical_up_pitch']=abs(pitch_up+90)<1
                checks['vertical_up_offcenter_hit']=upper is not None
                bridge.apply(AgentAction(look_dy=1200,ticks=6),request_surface_objects=True)
                down_view=bridge.surface_objects()
                pitch_down=bridge.packet.player_pitch
                checks['vertical_down_pitch']=abs(pitch_down-90)<1
                checks['vertical_down_noncollapsed_grid']=len(down_view.objects)>1
                vertical=dict(pitch_up=pitch_up,pitch_down=pitch_down,
                              up_hit_position=upper.xyz if upper else None,
                              up_hit_rays=upper.ray_indices if upper else [],
                              up_object_count=len(up_view.objects),
                              down_object_count=len(down_view.objects),
                              fine_grid=[up_view.columns,up_view.rows])
                print('VERTICAL SENSOR',json.dumps(vertical),flush=True)
            report=dict(tag=args.run_tag,fixture=xyz,profile=profile,checks=checks,
                        combined=combined,vertical=vertical,
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
