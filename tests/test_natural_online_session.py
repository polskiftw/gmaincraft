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
