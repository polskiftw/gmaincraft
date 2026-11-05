"""Versioned regressions for case randomization, observation, reward, replay."""
import pathlib,sys,unittest
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments'))
from gmaincraft.numeric_item_pickup_v2 import (
    SCHEMA,CASE_SCHEMA,FEATURE_NAMES,DirectionalCase,directional_cases_for_seed,
    egocentric_offset,EgocentricPickupMemory,ReplayPickupQ,progress_reward)
from gmaincraft.mcio_visible_items import SCHEMA as ITEM_SCHEMA,VisibleItemsV1,VisibleDroppedItemV1
from gmaincraft.player_knowledge import SCHEMA as PLAYER_SCHEMA,PlayerKnowledgeV1
from train_numeric_item_pickup import parse_args,metadata


def player(x=0.,z=0.,yaw=0.):
    return PlayerKnowledgeV1(PLAYER_SCHEMA,(x,70.,z),yaw,26.,20.,(),(),(),())


def eye(dx=0.,dz=3.):
    item=VisibleDroppedItemV1('minecraft:oak_log',(dx,70.,dz),
            float(np.hypot(dx,dz)),(.5,.5),5)
    return VisibleItemsV1(ITEM_SCHEMA,24.,(item,))


class PickupV2Tests(unittest.TestCase):
    def test_yaw_relative_movement_and_rotation_invariance(self):
        self.assertAlmostEqual(egocentric_offset(0.,3.,0.)[0],3.)
        self.assertAlmostEqual(egocentric_offset(-3.,0.,90.)[0],3.)
        self.assertAlmostEqual(egocentric_offset(-3.,0.,0.)[1],3.)
        a=EgocentricPickupMemory().observe(player(),eye(),elapsed_ticks=0)
        b=EgocentricPickupMemory().observe(player(yaw=90.),eye(-3.,0.),elapsed_ticks=0)
        self.assertAlmostEqual(a.features[2],b.features[2],places=5)
        self.assertAlmostEqual(a.features[3],b.features[3],places=5)
        self.assertEqual(len(a.features),len(FEATURE_NAMES))
        self.assertFalse(a.features.flags.writeable)
        gone=EgocentricPickupMemory()
        gone.observe(player(),eye(),elapsed_ticks=0)
        stale=gone.observe(player(x=1.),VisibleItemsV1(ITEM_SCHEMA,24.,()),elapsed_ticks=6)
        self.assertFalse(stale.item_seen)
        self.assertTrue(stale.features[1])
        self.assertGreater(stale.features[9],0.)

    def test_real_random_world_bearings_and_no_staged_target_in_observation(self):
        a=directional_cases_for_seed(812371,500)
        self.assertEqual(a,directional_cases_for_seed(812371,500))
        self.assertNotEqual(a,directional_cases_for_seed(812372,500))
        self.assertLess(min(x.yaw for x in a),-150.)
        self.assertGreater(max(x.yaw for x in a),150.)
        directions=[egocentric_offset(x.dx,x.dz,x.yaw) for x in a]
        self.assertGreater(sum(front<0 for front,_ in directions),5)
        self.assertGreater(sum(right<0 for _,right in directions),100)
        self.assertGreater(sum(right>0 for _,right in directions),100)
        arg=parse_args(['--run-tag','v2test','--kit','egoreplay_v2'])
        self.assertEqual(metadata(arg)['schema'],SCHEMA)
        self.assertEqual(metadata(arg)['case_contract'],CASE_SCHEMA)
        self.assertNotIn('world_xyz',metadata(arg)['observation'])

    def test_potential_reward_and_no_shaping_loop(self):
        toward=progress_reward(4.,3.,6,inventory_gain=0,failed=False)
        away=progress_reward(3.,4.,6,inventory_gain=0,failed=False)
        self.assertGreater(toward,0)
        self.assertLess(away,0)
        self.assertLess(toward+away,0)
        self.assertGreater(progress_reward(4.,4.,6,inventory_gain=1,failed=False),20.)
        with self.assertRaises(ValueError):progress_reward(4.,3.,4,inventory_gain=0,failed=False)

    def test_actual_replay_update_and_frozen_resume(self):
        actor=ReplayPickupQ(seed=31)
        old=EgocentricPickupMemory().observe(player(),eye(),elapsed_ticks=0)
        new=EgocentricPickupMemory().observe(player(z=1.),eye(),elapsed_ticks=6)
        before=actor.net.layers[0].weight.detach().clone()
        rows=[(old.features,0,2.,6,new.features,False)]*64
        metrics=actor.update(rows)
        self.assertEqual(metrics['buffer'],64)
        self.assertEqual(metrics['updates'],12)
        self.assertFalse(torch.equal(before,actor.net.layers[0].weight))
        success=[(old.features,0,25.,6,new.features,True)]
        actor.update(success)
        self.assertGreater(actor.state()['success_replay'].__len__(),0)
        frozen=ReplayPickupQ(seed=44)
        frozen.restore(actor.state(),optimizer=False)
        frozen.training=False
        self.assertEqual(frozen.act(old,greedy=True),actor.act(old,greedy=True))
        resumed=ReplayPickupQ(seed=44)
        resumed.restore(actor.state())
        self.assertEqual(resumed.episodes_seen,actor.episodes_seen)
        self.assertEqual(resumed.updates,actor.updates)
        self.assertEqual(resumed.act(old),actor.act(old))
        with self.assertRaises(ValueError):actor.update([(old.features,99,0.,6,new.features,False)])


if __name__=='__main__':unittest.main()
