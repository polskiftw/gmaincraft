"""Scratch pickup actor sees only own player state and current/stale item eyes."""
import pathlib,sys,unittest
import numpy as np

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.mcio_visible_items import (
    SCHEMA as ITEM_SCHEMA,VisibleItemsV1,VisibleDroppedItemV1)
from gmaincraft.player_knowledge import PlayerKnowledgeV1,SCHEMA as PLAYER_SCHEMA
from gmaincraft.numeric_item_pickup import (
    VisiblePickupMemory,NumericPickupActor,ACTIONS,motor_action,FEATURE_NAMES)


def player(x=0.,yaw=0.,logs=0):
    from gmaincraft.player_knowledge import OwnedItem
    return PlayerKnowledgeV1(PLAYER_SCHEMA,(x,70.,0.),yaw,26.,20.,
                             (OwnedItem(0,'minecraft:oak_log',logs),) if logs else (),
                             (),(),())


def items(*records):
    return VisibleItemsV1(ITEM_SCHEMA,24.,records)


def oak():
    return VisibleDroppedItemV1('minecraft:oak_log',(1.,70.,3.),3.2,(.65,.56),3)


class NumericPickupTests(unittest.TestCase):
    def test_normal_controls_only(self):
        for i in range(len(ACTIONS)):
            action=motor_action(i)
            self.assertNotIn('commands',vars(action))
            self.assertGreaterEqual(action.ticks,1)
            self.assertLessEqual(action.ticks,24)
        with self.assertRaises(ValueError):motor_action(len(ACTIONS))

    def test_own_memory_and_missing_visibility(self):
        memory=VisiblePickupMemory()
        zero=memory.observe(player(),items(),elapsed_ticks=0)
        self.assertEqual(zero.features.shape,(len(FEATURE_NAMES),))
        self.assertFalse(zero.item_seen)
        self.assertIsNone(zero.distance_to_remembered)
        seen=memory.observe(player(),items(oak()),elapsed_ticks=6)
        self.assertTrue(seen.item_seen)
        self.assertEqual(memory.last_item_id,'minecraft:oak_log')
        self.assertGreater(seen.features[1],0)
        away=memory.observe(player(x=.5),items(),elapsed_ticks=12)
        self.assertFalse(away.item_seen)
        self.assertTrue(away.features[1])
        self.assertAlmostEqual(away.distance_to_remembered,
                               np.sqrt(.5**2+3**2),places=4)
        expired=memory.observe(player(),items(),elapsed_ticks=610)
        self.assertFalse(expired.features[1])
        self.assertIsNone(expired.distance_to_remembered)
        self.assertFalse(expired.features.flags.writeable)
        multi=memory.observe(player(),items(oak(),oak()),elapsed_ticks=612)
        self.assertTrue(multi.item_seen)
        self.assertEqual(multi.visible_candidate_count,2)
        self.assertEqual(memory.last_item_id,'minecraft:oak_log')

    def test_from_scratch_actor_weight_update_and_frozen_restore(self):
        brain=NumericPickupActor(seed=42,device='cpu')
        obs=VisiblePickupMemory().observe(player(),items(oak()),elapsed_ticks=0)
        before=brain.net.actor.weight.detach().clone()
        rows=[(obs.features,0,-.2,6),(obs.features,1,.3,12),
              (obs.features,3,15.,24)]
        m=brain.update(rows)
        self.assertTrue(np.isfinite(m['loss']))
        self.assertEqual(brain.updates,1)
        self.assertFalse(brain.net.actor.weight.equal(before))
        state=brain.state()
        restored=NumericPickupActor(seed=777,device='cpu')
        restored.restore(state,optimizer=False)
        self.assertEqual(restored.act(obs,greedy=True),brain.act(obs,greedy=True))
        self.assertEqual(restored.updates,brain.updates)
        with self.assertRaises(ValueError):
            brain.update([(obs.features,99,0.,6)])
        with self.assertRaises(ValueError):
            brain.update([(obs.features,0,0.,5)])


if __name__=='__main__':unittest.main()
