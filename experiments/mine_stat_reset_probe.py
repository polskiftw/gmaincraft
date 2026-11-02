"""Prove event counts are per-episode, not cumulative MCio StatsUpdate jumps.

Two actual vanilla mining cases separated by one no-attack case. Same
controller-placed oak fixture is reset between episodes, no admin commands
during scored input. Stats counters are private evaluator signals, never
observation features. Does not train a model.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.agent_interface import MCioAgentBridge,natural_log_count
from gmaincraft.joint_curriculum import LogCase
from gmaincraft.vision_factorized import action_for_factors
from joint_log_ppo import launch_world


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-tag',required=True)
    p.add_argument('--visible',action='store_true')
    args=p.parse_args()
    args.frame_width=320;args.frame_height=180
    def trial(arena):
        for episode,attack in enumerate((True,False,True),1):
            obs=arena.reset(LogCase(0.,0.,0,2,0.,29.25))
            bridge=MCioAgentBridge(arena.client.ctrl,obs)
            raw_start=bridge.stat_total('minecraft:mined','minecraft:oak_log')
            arena.mined_events.begin_episode(raw_start)
            events=0
            for tick in range(20):
                bridge.apply(action_for_factors((0,0,int(attack))))
                events+=arena.mined_events.observe(
                    bridge.stat_total('minecraft:mined','minecraft:oak_log'))
            wood=natural_log_count(bridge.packet)
            raw_end=bridge.stat_total('minecraft:mined','minecraft:oak_log')
            print(f'EVENT AUDIT episode={episode} attack={attack} '
                  f'raw_start={raw_start} raw_end={raw_end} '
                  f'tracked_new_blocks={events} inventory_wood={wood}',flush=True)
            if attack and events!=1:
                raise RuntimeError('Expected exactly 1 new real mined oak log in attack episode')
            if not attack and events!=0:
                raise RuntimeError('False mined-event reward in no-attack negative control')
        print('PASS: cumulative MCio stat is decontaminated across resets',flush=True)
    launch_world(f'MDL_MineStats_{args.run_tag}',args,trial)

if __name__=='__main__':main()
