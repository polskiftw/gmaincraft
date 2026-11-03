"""Actual vanilla crack-stage feedback during Minecraft log mining.

Controller commands stage the disposable flat fixture ONLY before a scored
phase. The monitored player then uses normal MCio mouse/button input. This
is a sensor verification, not a learned speedrun or RL training episode.
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
from gmaincraft.agent_interface import AgentAction,MCioAgentBridge,natural_log_count
from gmaincraft.instance_options import temporary_game_options
from gmaincraft.mcio_status import ENV_NAME as STATUS_ENV
from gmaincraft.mcio_mining_feedback import (ENV_NAME as MINING_ENV,
                                              MiningFeedbackOption)
from gmaincraft.vision_settings import ensure_clean_visual_options
from mcio_throughput_probe import action,setup,reset

MCIO=ROOT/'mcio-data'
INSTANCE=MCIO/'instances'/'DemoInstance'


def snapshot(bridge,label):
    status=bridge.player_status()
    feedback=bridge.mining_feedback()
    target=status.crosshair_block
    result=dict(label=label,sequence=bridge.packet.last_action_sequence,
                frame=bridge.packet.frame_sequence,
                block=target.block_id if target else None,
                position=target.xyz if target else None,
                active=feedback.actively_destroying,
                stage=feedback.crack_stage,
                target_flag=feedback.crosshair_on_block,
                mined=bridge.stat_total('minecraft:mined','minecraft:oak_log'),
                picked=bridge.stat_total('minecraft:picked_up','minecraft:oak_log'),
                inventory=natural_log_count(bridge.packet))
    print('MINING OBS',json.dumps(result),flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-tag',required=True)
    parser.add_argument('--visible',action='store_true')
    args=parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,31}',args.run_tag):
        parser.error('Invalid unique tag')
    out=ROOT/'logs'/f'mcio_mining_feedback_{args.run_tag}.json'
    name='MDL_MiningFeedback_'+args.run_tag
    if out.exists() or (INSTANCE/'saves'/name).exists():
        parser.error('Refusing to overwrite saved world or prior results')
    world.WorldManager(mcio_dir=MCIO).copy('storage','DemoWorld','DemoInstance',name,force=False)
    ensure_clean_visual_options(INSTANCE)
    settings={'renderDistance':'4','enableVsync':'false','maxFps':'260'}
    options=types.RunOptions(mcio_dir=MCIO,instance_name='DemoInstance',world_name=name,
          mcio_mode=types.MCioMode.SYNC,hide_window=not args.visible,
          gpu_lib=None if args.visible else 'nvidia',width=320,height=180,
          env_extra={STATUS_ENV:'1',MINING_ENV:'1'})
    with temporary_game_options(INSTANCE,settings,run_tag=args.run_tag):
        launcher=instance.Launcher(options)
        ctrl=None
        try:
            t0=time.perf_counter()
            launcher.launch(wait=False)
            ctrl=controller.ControllerSync(action_port=options.action_port,
                    observation_port=options.observation_port,connection_timeout=65)
            xyz=setup(ctrl)
            reset(ctrl,xyz)
            time.sleep(.15)  # real server /setblock settling time
            obs=None
            manager=env_util.InputStateManager()
            for _ in range(15): obs,_,_=action(ctrl,manager)
            if obs.get_option(MiningFeedbackOption) is None:
                raise AssertionError('New opt-in feedback was not sent by Fabric')
            bridge=MCioAgentBridge(ctrl,obs)
            print('MINING READY',json.dumps(dict(tag=args.run_tag,fixture=xyz,
                 startup_s=round(time.perf_counter()-t0,3))),flush=True)
            checks={}
            samples=[]
            bridge.apply(AgentAction(ticks=6))
            idle=snapshot(bridge,'idle_log_visible');samples.append(idle)
            checks['idle_inactive']=not idle['active'] and idle['stage']==-1
            checks['idle_can_target_oak']=idle['block']=='minecraft:oak_log' and idle['target_flag']
            baseline=idle['mined']
            # Click for four short decisions, release, and then sustain for
            # longer. No admin commands or controller geometry during play.
            for idx in range(4):
                bridge.apply(AgentAction(keys=('LEFT_BUTTON',),ticks=6))
                samples.append(snapshot(bridge,f'initial_hold_{idx+1}'))
            bridge.apply(AgentAction(ticks=6))
            released=snapshot(bridge,'released');samples.append(released)
            checks['release_resets_progress']=not released['active'] and released['stage']==-1
            for idx in range(20):
                bridge.apply(AgentAction(keys=('LEFT_BUTTON',),ticks=6))
                row=snapshot(bridge,f'sustained_hold_{idx+1}')
                samples.append(row)
                if row['mined']>baseline:break
            checks['active_stage_sampled']=any(s['active'] and s['stage']>=0 for s in samples)
            checks['nonzero_crack_stage']=any(s['active'] and s['stage']>0 for s in samples)
            checks['actual_mined_event']=bridge.stat_total('minecraft:mined','minecraft:oak_log')>baseline
            # An ordinary forward pickup attempt can finish mining but is
            # not a required sensor property. Retain as separate telemetry.
            for _ in range(8):
                if natural_log_count(bridge.packet):break
                bridge.apply(AgentAction(keys=('W',),ticks=6))
            finished=snapshot(bridge,'post_approach');samples.append(finished)
            bridge.apply(AgentAction(look_dx=850,ticks=6))
            turned=snapshot(bridge,'turned_away');samples.append(turned)
            checks['turned_away_inactive']=not turned['active'] and turned['stage']==-1
            checks['turned_away_not_oak']=turned['block']!='minecraft:oak_log'
            checks['sequence_advances']=all(b['sequence']>a['sequence'] for a,b in zip(samples,samples[1:]))
            report=dict(tag=args.run_tag,world=name,fixture=xyz,settings=settings,
                        game_time_s=round(time.perf_counter()-t0,3),
                        checks=checks,samples=samples,
                        actual_vanilla_inventory_logs=finished['inventory'],
                        controller_commands_during_scored_phase=0,
                        learned_agent=False)
            out.parent.mkdir(parents=True,exist_ok=True)
            with out.open('x',encoding='utf8') as handle:
                json.dump(report,handle,indent=2);handle.write('\n')
            print('MINING AUDIT',json.dumps(checks,sort_keys=True),'all_passed',all(checks.values()),flush=True)
            print('SAVED',out,flush=True)
            if not all(checks.values()):
                raise AssertionError('Live Minecraft mining feedback control failed')
        finally:
            if ctrl is not None:
                try:ctrl.send_stop()
                except Exception as exc:print('STOP WARNING',repr(exc),flush=True)
                ctrl.close()
            launcher.close()

if __name__=='__main__':main()
