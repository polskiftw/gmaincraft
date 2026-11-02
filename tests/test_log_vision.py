"""Synthetic RGB-only tests, runnable in CI without a Minecraft installation."""
import pathlib
import sys
import unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.log_vision import visual_features,FEATURE_COUNT,VisualAimPolicy


class VisionFeatureTests(unittest.TestCase):
    def test_only_rgb_correct_feature_length(self):
        image=np.full((240,320,3),[140,194,250],dtype=np.uint8)
        v=visual_features(image)
        self.assertEqual(v.shape,(FEATURE_COUNT,))
        self.assertTrue(np.isfinite(v).all())
        self.assertEqual(v[-1],1)

    def test_left_and_right_pixel_content_distinguishable(self):
        bg=np.full((240,320,3),[140,194,250],dtype=np.uint8)
        left=bg.copy();left[90:160,30:80]=[130,95,47]
        right=bg.copy();right[90:160,230:280]=[130,95,47]
        self.assertLess(visual_features(left)[-4],visual_features(right)[-4])

    def test_up_and_down_pixel_content_distinguishable(self):
        bg=np.full((240,320,3),[140,194,250],dtype=np.uint8)
        up=bg.copy();up[20:60,100:170]=[130,95,47]
        down=bg.copy();down[130:170,100:170]=[130,95,47]
        self.assertLess(visual_features(up)[-3],visual_features(down)[-3])

    def test_non_rgb_rejected(self):
        with self.assertRaises(ValueError):visual_features(np.zeros((50,50),dtype=np.uint8))

    def test_random_weights_and_valid_probabilities(self):
        v=np.ones(FEATURE_COUNT,dtype=np.float64)
        p=VisualAimPolicy(seed=101);q=VisualAimPolicy(seed=102)
        self.assertFalse(np.array_equal(p.weights,q.weights))
        ps=p.probabilities(v)
        self.assertEqual(len(ps),5)
        self.assertAlmostEqual(ps.sum(),1.0)

    def test_reward_changes_policy_from_random_weights(self):
        v=np.ones(FEATURE_COUNT,dtype=np.float64)
        p=VisualAimPolicy(seed=101)
        previous=p.weights.copy()
        p.learn(v,0,p.probabilities(v),4)
        self.assertFalse(np.array_equal(p.weights,previous))
        self.assertEqual(p.updates,1)


if __name__=='__main__':unittest.main()
