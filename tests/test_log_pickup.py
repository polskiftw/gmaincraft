"""Tests for the RGB-only collection-stage feature and stochastic policy."""
import pathlib
import sys
import unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.log_pickup import pickup_features,PickupPolicy,PICKUP_FEATURES,fixture_item_command

class PickupTests(unittest.TestCase):
    def test_rgb_contract(self):
        im=np.full((360,640,3),[120,170,220],dtype=np.uint8)
        x=pickup_features(im)
        self.assertEqual(x.shape,(PICKUP_FEATURES,))
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(x[-1],1.0)
        with self.assertRaises(ValueError):pickup_features(np.zeros((15,15),dtype=np.uint8))

    def test_pixel_location_moves_center_feature(self):
        im=np.full((360,640,3),[120,170,220],dtype=np.uint8)
        left=im.copy();right=im.copy()
        left[160:190,120:170]=[130,90,50]
        right[160:190,440:490]=[130,90,50]
        a=pickup_features(left);b=pickup_features(right)
        self.assertLess(a[-4],b[-4]);self.assertGreater(abs(b[-4]-a[-4]),.6)
        self.assertNotEqual(int(a[:7].argmax()),int(b[:7].argmax()))

    def test_fixture_is_reproducible_zero_initial_velocity(self):
        cmd=fixture_item_command(1.5,66.2,3.7)
        self.assertIn('minecraft:oak_log',cmd)
        self.assertIn('Motion:[0.0d,0.0d,0.0d]',cmd)
        self.assertIn('PickupDelay:0s',cmd)
        with self.assertRaises(ValueError):fixture_item_command(float('nan'),1,2)

    def test_model_starts_random_and_learns_scalar_reward(self):
        p=PickupPolicy(seed=23);q=PickupPolicy(seed=24)
        self.assertFalse(np.array_equal(p.weights,q.weights))
        features=np.ones(PICKUP_FEATURES)
        pr=p.probabilities(features)
        self.assertAlmostEqual(float(pr.sum()),1.0)
        before=p.weights.copy();p.learn(features,1,pr,2.2)
        self.assertFalse(np.array_equal(before,p.weights))
        self.assertEqual(p.updates,1)

if __name__=='__main__':unittest.main()
