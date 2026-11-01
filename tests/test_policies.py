"""Fast, Minecraft-independent unit tests of the prototype policies."""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
from navigation_learner import LinearPolicy
from orientation_learner import EgocentricPolicy, angular_difference


class PolicyMathTests(unittest.TestCase):
    def test_absolute_goal_features_are_unit_length(self):
        features = LinearPolicy.features(0, 0, (3, 4))
        np.testing.assert_allclose(features, [0.6, 0.8, 1.0])

    def test_probabilities_normalize(self):
        policy = LinearPolicy(seed=17)
        probs = policy.probabilities(np.array([1.0, 0.0, 1.0]))
        self.assertEqual(probs.shape, (4,))
        self.assertTrue(np.all(probs > 0))
        self.assertAlmostEqual(probs.sum(), 1.0)

    def test_training_step_changes_random_weights(self):
        policy = LinearPolicy(seed=17)
        features = LinearPolicy.features(0, 0, (2, 0))
        probs = policy.probabilities(features)
        before = policy.weights.copy()
        policy.learn(features, 0, probs, 1.0)
        self.assertFalse(np.array_equal(policy.weights, before))
        self.assertEqual(policy.updates, 1)

    def test_camera_relative_forward_direction(self):
        # Minecraft yaw=0 faces +Z; yaw=90 faces -X.
        np.testing.assert_allclose(EgocentricPolicy.features(0, 0, (0, 5), 0), [1, 0, 1], atol=1e-12)
        np.testing.assert_allclose(EgocentricPolicy.features(0, 0, (-5, 0), 90), [1, 0, 1], atol=1e-12)
        np.testing.assert_allclose(EgocentricPolicy.features(0, 0, (0, -5), 180), [1, 0, 1], atol=1e-12)

    def test_yaw_wraparound(self):
        self.assertAlmostEqual(angular_difference(179, -179), 2.0)
        self.assertAlmostEqual(angular_difference(0, 360), 0.0)


if __name__ == "__main__":
    unittest.main()
