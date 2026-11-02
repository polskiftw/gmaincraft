"""New variable-tree scenarios remain isolated from the policy code."""
import math
import pathlib
import sys
import unittest

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.joint_curriculum import sample_cases
from gmaincraft.vision_ppo import ACTION_COUNT, action_for_index


class JointCurriculumTests(unittest.TestCase):
    def test_randomized_player_tree_camera_cases(self):
        a=sample_cases(83742,200)
        self.assertEqual(a,sample_cases(83742,200))
        self.assertNotEqual(a,sample_cases(83743,200))
        self.assertGreater(len({(p.tree_dx,p.tree_dz) for p in a}),20)
        self.assertGreater(len({(p.player_dx,p.player_dz) for p in a}),150)
        self.assertTrue(all(2.3<=math.hypot(p.tree_dx-p.player_dx,p.tree_dz-p.player_dz)<=7.3
                            for p in a))

    def test_action_macro_maps_to_genuine_keys_and_bounded_timing(self):
        self.assertEqual(ACTION_COUNT,11)
        for i in range(ACTION_COUNT):
            action=action_for_index(i,3)
            self.assertGreaterEqual(action.ticks,1)
            self.assertLessEqual(action.ticks,3)
            self.assertFalse(hasattr(action,'commands'))
        self.assertEqual(action_for_index(8).keys,('LEFT_BUTTON',))
        self.assertEqual(action_for_index(8).ticks,80)
        with self.assertRaises(ValueError): action_for_index(-1)
        with self.assertRaises(ValueError): action_for_index(ACTION_COUNT)

    def test_count_validation(self):
        with self.assertRaises(ValueError): sample_cases(4,0)

if __name__=='__main__': unittest.main()
