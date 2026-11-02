"""Vanilla mining sanity test: 6-tick button holds across separate actions.

A privileged controller creates a centered log fixture and initial aim ONLY
before scoring. The scored sequence is 120 ticks of ordinary LEFT_BUTTON in
20 separate 6-tick actions, then an ordinary forward walk for pickup. It is
NOT a learned policy; the test establishes that factorized attack controls
can hold a real button continuously long enough to mine a real block.
"""
import argparse
from types import SimpleNamespace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.agent_interface import MCioAgentBridge,natural_log_count
from gmaincraft.vision_factorized import action_for_factors
from gmaincraft.joint_curriculum import LogCase
from joint_log_ppo import launch_world


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-tag',required=True)
    p.add_argument('--visible',action='store_true')
    args=p.parse_args()
    args.frame_width=320;args.frame_height=180
    def trial(arena):
        ob=arena.reset(LogCase(0.,0.,0,2,0.,29.25))
        bridge=MCioAgentBridge(arena.client.ctrl,ob)
        assert not natural_log_count(bridge.packet)
        baseline_mined=bridge.stat_total('minecraft:mined','minecraft:oak_log')
        baseline_pickup=bridge.stat_total('minecraft:picked_up','minecraft:oak_log')
        distance,angle=arena.geometry(ob)
        print(f'FIXTURE initial distance={distance:.3f} angular_error={angle:.3f}',flush=True)
        for i in range(20):
            bridge.apply(action_for_factors((0,0,1)))
        print(f'AFTER 120 TICKS left held: wood_inventory={natural_log_count(bridge.packet)} '
              f'mined_delta={bridge.stat_total("minecraft:mined","minecraft:oak_log")-baseline_mined}',flush=True)
        for i in range(8):
            bridge.apply(action_for_factors((1,0,0)))
            if natural_log_count(bridge.packet): break
        result=natural_log_count(bridge.packet)
        print(f'VANILLA LOG ACQUISITION={result}; approach_steps={i+1}; pos={bridge.packet.player_pos} '
              f'mined_delta={bridge.stat_total("minecraft:mined","minecraft:oak_log")-baseline_mined} '
              f'picked_up_delta={bridge.stat_total("minecraft:picked_up","minecraft:oak_log")-baseline_pickup}',flush=True)
        if result < 1:raise RuntimeError('No wood acquired after 120-tick held attack and ordinary approach')
        return result
    launch_world('MDL_ContinuousMineProbe_'+args.run_tag,args,trial)
    print('PASS: repeated short actions preserve vanilla continuous attack',flush=True)

if __name__=='__main__':main()
