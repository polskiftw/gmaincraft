"""Memory curiosity and monotonic events on real-player-shaped observations."""
from collections import defaultdict
import pathlib
import sys
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.natural_online_session import NaturalSessionMemory
from gmaincraft.player_knowledge import PlayerKnowledgeV1, OwnedItem
from gmaincraft.mcio_surface_objects import SurfaceObjectsV1, ProjectedBlockV1
from gmaincraft.mcio_status import CrosshairBlock
from gmaincraft.wood_target_reward import WoodTargetFirstContact,WoodEngagement,WoodSustainedAttack


def player(x=5.5,z=0.5,logs=0):
    return PlayerKnowledgeV1('player_knowledge_v1',(x,70,z),0.,0.,20.,
        (OwnedItem(0,'minecraft:oak_log',logs),) if logs else (),(),(),())


def scene():
    return SurfaceObjectsV1('gmaincraft_surface_objects_v1',13,9,70.,64.,(
        ProjectedBlockV1('minecraft:oak_log',(6,71,4),('west',),3.,15,0,
            (30,31),2/117,(.44,.4),(.4,.35,.5,.45)),))


class NaturalSessionTests(unittest.TestCase):
    def test_first_hit_and_world_owned_memory_not_award_same_place_twice(self):
        memory=NaturalSessionMemory()
        initial=defaultdict(dict)
        memory.ensure_world(0,player(),initial)
        first=memory.accrue(0,player(),scene(),initial,(1,2,0),6)
        second=memory.accrue(0,player(),scene(),initial,(0,0,0),6)
        self.assertEqual((first['fresh_cell'],first['new_visible_surfaces']),(1,1))
        self.assertEqual((second['fresh_cell'],second['new_visible_surfaces']),(0,0))
        self.assertGreater(first['reward'],second['reward'])
        self.assertEqual(memory.ticks[0],12)
        self.assertEqual(memory.cell_visits(0,player()),2)
        memory.ensure_world(1,player(),initial)
        fresh_other=memory.accrue(1,player(),scene(),initial,(1,0,0),6)
        self.assertEqual(fresh_other['new_visible_surfaces'],1)
        self.assertEqual(fresh_other['fresh_cell'],1)
        saved=memory.state()
        restored=NaturalSessionMemory().restore(saved)
        self.assertEqual(restored.state()['ticks'],{0:12,1:6})
        self.assertEqual(restored.cell_visits(0,player()),2)
        self.assertEqual(len(restored.surfaces[0]),1)

    def test_versioned_visible_wood_sighting_reward_not_repeated(self):
        old=NaturalSessionMemory()
        new=NaturalSessionMemory()
        for memory in (old,new):memory.ensure_world(0,player(),{})
        v1=old.accrue(0,player(),scene(),{},(0,0,0),6,reward_profile='explore_v1')
        v2=new.accrue(0,player(),scene(),{},(0,0,0),6,reward_profile='wood_discovery_v2')
        self.assertEqual(v2['new_visible_wood'],1)
        self.assertEqual(v2['visible_wood_count'],1)
        self.assertAlmostEqual(v2['reward']-v1['reward'],.32)
        repeat=new.accrue(0,player(),scene(),{},(0,0,0),6,reward_profile='wood_discovery_v2')
        self.assertEqual(repeat['new_visible_wood'],0)
        self.assertEqual(repeat['intrinsic_reward'],0.)
        with self.assertRaises(ValueError):
            new.accrue(0,player(),scene(),{},(0,0,0),6,reward_profile='unknown')

    def test_v5_reward_rebalances_real_wood_over_incidental_nonwood_blocks(self):
        old=NaturalSessionMemory();new=NaturalSessionMemory()
        for memory in (old,new):memory.ensure_world(0,player(),{})
        base=old.accrue(0,player(),scene(),{},(1,0,1),6,reward_profile='wood_discovery_v2')
        changed=new.accrue(0,player(),scene(),{},(1,0,1),6,reward_profile='wood_target_v5')
        self.assertEqual(base['new_visible_wood'],changed['new_visible_wood'])
        self.assertLess(changed['reward'],base['reward']) # curiosity is no longer dominant
        stats={'minecraft:mined':{'minecraft:oak_log':1,'minecraft:stone':1}}
        gained=new.accrue(0,player(logs=1),scene(),stats,(0,0,1),6,
                          reward_profile='wood_target_v5')
        self.assertEqual(gained['new_wood_mined'],1)
        self.assertEqual(gained['wood_inventory_gain'],1)
        self.assertGreater(gained['extrinsic_reward'],25.)
        # Never mint the same stat/inventory reward on a repeat sample.
        repeat=new.accrue(0,player(logs=1),scene(),stats,(0,0,1),6,
                          reward_profile='wood_target_v5')
        self.assertEqual(repeat['extrinsic_reward'],0.)

    def test_v5_wood_target_reward_once_from_typed_crosshair_only(self):
        monitor=WoodTargetFirstContact(max_blocks=2,bonus=.9)
        oak=CrosshairBlock('minecraft:oak_log',(6,71,4),'west',3.)
        birch=CrosshairBlock('minecraft:birch_log',(7,71,4),'west',3.)
        stone=CrosshairBlock('minecraft:stone',(8,71,4),'west',3.)
        self.assertEqual(monitor.observe(None),0.)
        self.assertEqual(monitor.observe(stone),0.)
        self.assertAlmostEqual(monitor.observe(oak),.9)
        self.assertEqual(monitor.observe(oak),0.)
        self.assertAlmostEqual(monitor.observe(birch),.9)
        self.assertEqual(monitor.observe(CrosshairBlock('minecraft:oak_log',(8,71,4),'west',3.)),0.)
        self.assertEqual(monitor.observe(oak),0.)
        with self.assertRaises(TypeError):monitor.observe(object())
        with self.assertRaises(ValueError):WoodTargetFirstContact(max_blocks=0)

    def test_v6_first_wood_attack_is_bounded_and_requires_actual_hit(self):
        engagement=WoodEngagement(max_blocks=1)
        oak=CrosshairBlock('minecraft:oak_log',(6,71,4),'west',3.)
        birch=CrosshairBlock('minecraft:birch_log',(7,71,4),'west',3.)
        self.assertEqual(engagement.observe(None,attack=True),(0.,0.))
        self.assertEqual(engagement.observe(oak,attack=False),(.6,0.))
        self.assertEqual(engagement.observe(oak,attack=True),(0.,1.8))
        self.assertEqual(engagement.observe(oak,attack=True),(0.,0.))
        self.assertEqual(engagement.observe(birch,attack=True),(0.,0.))
        with self.assertRaises(TypeError):engagement.observe(oak,attack=1)
        with self.assertRaises(ValueError):WoodEngagement(max_blocks=0)

    def test_v6_reward_values_real_pickup_far_above_passive_surface_sighting(self):
        memory=NaturalSessionMemory();memory.ensure_world(0,player(),{})
        first=memory.accrue(0,player(),scene(),{},(0,0,0),6,
                            reward_profile='wood_engagement_v6')
        self.assertEqual(first['new_visible_wood'],1)
        self.assertLess(first['intrinsic_reward'],.2)
        changed=memory.accrue(0,player(logs=1),scene(),
                              {'minecraft:mined':{'minecraft:oak_log':1}},(0,0,1),6,
                              reward_profile='wood_engagement_v6')
        self.assertGreater(changed['extrinsic_reward'],50.)
        again=memory.accrue(0,player(logs=1),scene(),
                            {'minecraft:mined':{'minecraft:oak_log':1}},(0,0,1),6,
                            reward_profile='wood_engagement_v6')
        self.assertEqual(again['extrinsic_reward'],0.)

    def test_v8_reward_sightings_no_longer_dominate_actual_wood_inventory(self):
        old=NaturalSessionMemory();new=NaturalSessionMemory()
        for memory in (old,new):memory.ensure_world(0,player(),{})
        v7=old.accrue(0,player(),scene(),{},(0,0,1),6,reward_profile='wood_target_v5')
        v8=new.accrue(0,player(),scene(),{},(0,0,1),6,reward_profile='wood_sustain_v8')
        self.assertEqual(v8['new_visible_wood'],1)
        self.assertLess(v8['reward'],v7['reward'])
        gained=new.accrue(0,player(logs=1),scene(),
                         {'minecraft:mined':{'minecraft:oak_log':1}},
                         (0,0,1),6,reward_profile='wood_sustain_v8')
        self.assertGreaterEqual(gained['extrinsic_reward'],57.)
        repeat=new.accrue(0,player(logs=1),scene(),
                         {'minecraft:mined':{'minecraft:oak_log':1}},
                         (0,0,1),6,reward_profile='wood_sustain_v8')
        self.assertEqual(repeat['extrinsic_reward'],0.)

    def test_v8_stable_wood_attack_reward_cannot_be_farmed_or_spoofed(self):
        tracker=WoodSustainedAttack(max_blocks=1,max_samples=3,bonus=.75)
        oak=CrosshairBlock('minecraft:oak_log',(4,70,6),'west',3.)
        other=CrosshairBlock('minecraft:birch_log',(5,70,6),'west',3.)
        self.assertEqual(tracker.observe(None,stationary_attack=True),0.)
        self.assertEqual(tracker.observe(oak,stationary_attack=False),0.)
        self.assertEqual([tracker.observe(oak,stationary_attack=True) for _ in range(5)],
                         [.75,.75,.75,0.,0.])
        self.assertEqual(tracker.observe(other,stationary_attack=True),0.)
        with self.assertRaises(TypeError):tracker.observe(oak,stationary_attack=1)
        with self.assertRaises(TypeError):tracker.observe(object(),stationary_attack=True)
        with self.assertRaises(ValueError):WoodSustainedAttack(max_samples=0)

    def test_cumulative_stats_stay_monotonic_across_bridge_restarts(self):
        memory=NaturalSessionMemory()
        memory.ensure_world(0,player(),{'minecraft:mined':{'minecraft:oak_log':3},
                                         'minecraft:picked_up':{'minecraft:oak_log':2}})
        stats={'minecraft:mined':{'minecraft:oak_log':4},
               'minecraft:picked_up':{'minecraft:oak_log':3}}
        step=memory.accrue(0,player(logs=1),scene(),stats,(0,0,1),6)
        self.assertEqual(step['new_wood_mined'],1)
        self.assertEqual(step['new_wood_picked_stat'],1)
        self.assertEqual(step['wood_inventory_gain'],1)
        # MCio bridge resets its incremental cache to zero: do not count as reward.
        memory.ensure_world(0,player(logs=1),{})
        stale=memory.accrue(0,player(logs=1),scene(),stats,(0,0,0),6)
        self.assertEqual(stale['new_mined_blocks'],0)
        self.assertEqual(stale['wood_inventory_gain'],0)
        self.assertEqual(stale['new_wood_picked_stat'],0)
        memory.ensure_world(0,player(logs=1),{})
        later=memory.accrue(0,player(logs=1),scene(),{'minecraft:mined':{'minecraft:oak_log':5}},
                            (0,0,1),6)
        self.assertEqual(later['new_wood_mined'],1)

if __name__=='__main__':unittest.main()
