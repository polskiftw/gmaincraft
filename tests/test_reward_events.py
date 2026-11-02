import pathlib
import sys
import unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.reward_events import mined_event_delta,mined_event_reward


class EventRewardTests(unittest.TestCase):
    def test_delta_and_cap(self):
        self.assertEqual(mined_event_delta(8,9),1)
        self.assertEqual(mined_event_reward(8,9,reward_per_block=5),5)
        self.assertEqual(mined_event_reward(8,8,reward_per_block=5),0)
        self.assertEqual(mined_event_reward(8,19,reward_per_block=5,cap=3),15)
        with self.assertRaises(ValueError):mined_event_delta(8,3)
        with self.assertRaises(ValueError):mined_event_reward(0,1,reward_per_block=float('nan'))
        with self.assertRaises(ValueError):mined_event_reward(0,1,cap=0)

if __name__=='__main__':unittest.main()


class CumulativeStatRaceTest(unittest.TestCase):
    def test_stat_cache_resets_to_zero_but_absolute_mined_is_cumulative(self):
        from gmaincraft.reward_events import CumulativeMinedTracker
        t=CumulativeMinedTracker()
        t.begin_episode(0)
        self.assertEqual(t.observe(0),0)
        self.assertEqual(t.observe(1),1)
        # Next Minecraft scenario uses a new MCio bridge: its local cache is
        # empty (0), then reports cumulative 1 on first update. No reward.
        t.begin_episode(0)
        self.assertEqual(t.observe(0),0)
        self.assertEqual(t.observe(1),0)
        self.assertEqual(t.observe(2),1)
        t.begin_episode(0)
        self.assertEqual(t.observe(0),0)
        self.assertEqual(t.observe(2),0)
        self.assertEqual(t.observe(3),1)
        self.assertEqual(t.absolute,3)

    def test_starting_from_nonzero_full_stat(self):
        from gmaincraft.reward_events import CumulativeMinedTracker
        t=CumulativeMinedTracker()
        with self.assertRaises(RuntimeError):t.observe(2)
        t.begin_episode(12)
        self.assertEqual(t.observe(12),0)
        self.assertEqual(t.observe(13),1)
        t.begin_episode(0)
        self.assertEqual(t.observe(0),0)
        self.assertEqual(t.observe(13),0)
        self.assertEqual(t.observe(14),1)
