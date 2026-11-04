"""Controlled Minecraft fixture to calibrate opt-in item-entity memory eyes.

This is a SENSOR LAB test, NOT agent training, policy input or an autonomous
pickup score. The setup controller stages a drop and glass/stone/leaf walls
using admin commands ONLY on a fresh copied disposable world. Each measured
observation uses a real synchronized vanilla idle action with an explicit
fresh item-sensor request. Ordinary no-request action must not leak stale
item information. The default RGB renderer is kept ON as a diagnostic
witness (not a policy input).
"""
import argparse
import json
import math
from pathlib import Path
import re
import sys
import time

from mcio_ctrl import controller,instance,types,world
from mcio_ctrl.envs import env_util

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from gmaincraft.agent_interface import AgentAction,MCioAgentBridge
from gmaincraft.instance_options import temporary_game_options
from gmaincraft.mcio_visible_items import (
    ENV_NAME, ON_DEMAND_ENV, VisibleItemsOption)
from gmaincraft.log_pickup import fixture_item_command
from gmaincraft.vision_settings import ensure_clean_visual_options
from mcio_throughput_probe import action,setup

MCIO=ROOT/'mcio-data'
INSTANCE=MCIO/'instances'/'DemoInstance'


def run(ctrl,xyz):
    x,y,z=xyz
    manager=env_util.InputStateManager()
    checks={}
    results=[]
    for label,block,away,with_item in (
        ('no_item','air',False,False),
        ('clear_visible','air',False,True),
        ('stone_wall','stone',False,True),
        ('glass_wall','glass',False,True),
        ('leaves_wall','oak_leaves',False,True),
        ('looking_away','air',True,True)):
        block_cmd=(f'fill {x-1} {y+1} {z+2} {x+1} {y+3} {z+2} minecraft:{block} replace'
                   if block!='air' else '')
        commands=[
            'kill @e[type=minecraft:item]',
            f'fill {x-3} {y+1} {z+1} {x+3} {y+6} {z+6} minecraft:air replace',
            f'tp @s {x+.5:.3f} {y+1.0:.3f} {z+.5:.3f} {180 if away else 0} 21',
            'clear @s',
            *([block_cmd] if block_cmd else []),
            *([fixture_item_command(x+.5,y+1.2,z+3.5)] if with_item else []),
        ]
        ob,_,_=action(ctrl,manager,commands=commands,clear=True)
        for _ in range(10):
            ob,_,_=action(ctrl,manager)
        # A separate action-only bridge requests a fresh, bounded item
        # observation. No controller world information is handed to it.
        bridge=MCioAgentBridge(ctrl,ob)
        if ob.get_option(VisibleItemsOption) is not None:
            raise AssertionError('Unrequested item sensor was already present')
        bridge.apply(AgentAction(ticks=3),request_visible_items=True,
                     return_view=False)
        detected=bridge.visible_items()
        raw=bridge.packet
        pixels=raw.get_frame()
        scene=dict(scene=label,reported_item_ids=[item.item_id for item in detected.items],
                   visible_item_count=len(detected.items),
                   samples=[item.visible_samples for item in detected.items],
                   camera_uv=[tuple(round(v,3) for v in item.screen_uv)
                              for item in detected.items],
                   range=[round(item.distance,3) for item in detected.items],
                   player_xyz=[round(v,3) for v in raw.player_pos],
                   rgb_valid=pixels.shape==(180,320,3),
                   frame_seq=raw.frame_sequence)
        results.append(scene)
        print('ITEM VISIBILITY',json.dumps(scene,sort_keys=True),flush=True)
        bridge.apply(AgentAction(ticks=1),return_view=False)
        checks[label+'_no_stale']=(
            bridge.packet.get_option(VisibleItemsOption) is None)
        checks[label+'_rgb']=scene['rgb_valid']
        checks[label+'_camera_advances']=bridge.packet.frame_sequence>raw.frame_sequence
        if label=='no_item':checks['empty_when_no_item']=len(detected.items)==0
        if label=='clear_visible':
            checks['visible_oak_in_fov']=any(e.item_id=='minecraft:oak_log'
                                              for e in detected.items)
        if label=='stone_wall':checks['no_xray_through_stone']=len(detected.items)==0
        if label=='looking_away':checks['no_backward_eyes']=len(detected.items)==0
    return checks,results


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--visible',action='store_true')
    args=ap.parse_args(argv)
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,27}',args.run_tag):
        ap.error('Invalid test tag')
    new_world='MDL_ItemEyes_'+args.run_tag
    dest=INSTANCE/'saves'/new_world
    out=ROOT/'logs'/f'mcio_visible_items_{args.run_tag}.json'
    if dest.exists() or out.exists():
        ap.error('Refusing to overwrite an existing world/report')
    world.WorldManager(mcio_dir=MCIO).copy('storage','DemoWorld',
                                         'DemoInstance',new_world,force=False)
    ensure_clean_visual_options(INSTANCE)
    opt=types.RunOptions(mcio_dir=MCIO,instance_name='DemoInstance',world_name=new_world,
           mcio_mode=types.MCioMode.SYNC,hide_window=not args.visible,
           gpu_lib=None if args.visible else 'nvidia',width=320,height=180,
           env_extra={ENV_NAME:'1',ON_DEMAND_ENV:'1'})
    options={'renderDistance':'4','enableVsync':'false','maxFps':'260'}
    with temporary_game_options(INSTANCE,options,run_tag=args.run_tag):
        launcher=instance.Launcher(opt)
        ctrl=None
        try:
            started=time.perf_counter()
            launcher.launch(wait=False)
            ctrl=controller.ControllerSync(action_port=opt.action_port,
                observation_port=opt.observation_port,connection_timeout=70)
            xyz=setup(ctrl)
            checks,scenes=run(ctrl,xyz)
            report=dict(tag=args.run_tag,fixture=True,source_world='DemoWorld',
                        controller_commands_only_during_lab_setup=True,
                        ordinary_actions_for_sensor=True,
                        player_model_not_connected=True,
                        valid=all(checks.values()),checks=checks,scenes=scenes,
                        runtime_s=round(time.perf_counter()-started,3))
            out.parent.mkdir(parents=True,exist_ok=True)
            with out.open('x') as f:
                json.dump(report,f,indent=2,sort_keys=True)
                f.write('\n')
            print('ITEM VISIBILITY SAVED',out,flush=True)
            print('ITEM VISIBILITY CHECKS',json.dumps(checks,sort_keys=True),flush=True)
            if not all(checks.values()):
                raise AssertionError('Visible item camera/occlusion contract failed')
        finally:
            if ctrl is not None:
                try:ctrl.send_stop()
                except Exception as exc:print('STOP WARNING',repr(exc),flush=True)
                ctrl.close()
            launcher.close()


if __name__=='__main__':main()
