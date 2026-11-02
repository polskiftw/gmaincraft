"""Staging tests for independent, non-clobbering orientation experiments."""
import pathlib, random, sys, unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'experiments'))
from orientation_learner import named_run_paths, make_scenario, EgocentricPolicy

class SeededOrientationTests(unittest.TestCase):
    def test_base_names_unchanged(self):
        world, log, model = named_run_paths('')
        self.assertEqual(world, 'MDL_OrientationLab_v1')
        self.assertEqual(log.name, 'orientation_training.csv')
        self.assertEqual(model.name, 'orientation_policy.npz')

    def test_tagged_names_independent(self):
        a = named_run_paths('rep_b')
        b = named_run_paths('rep_c')
        self.assertNotEqual(a,b)
        self.assertNotEqual(a[2], named_run_paths('')[2])

    def test_invalid_tags_rejected(self):
        for tag in ['../home', '/root', 'a/b', '..', '.hidden', '-flag', 'a'*41, 'a b']:
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                named_run_paths(tag)

    def test_scenario_sampling_repeatable_and_distinct(self):
        class FakeArena: cx=0.5; cz=0.5
        a = make_scenario(FakeArena, random.Random(81003))
        b = make_scenario(FakeArena, random.Random(81003))
        c = make_scenario(FakeArena, random.Random(81004))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_different_policy_seeds_change_weights(self):
        import numpy as np
        a=EgocentricPolicy(seed=2718)
        b=EgocentricPolicy(seed=2719)
        self.assertFalse(np.array_equal(a.weights,b.weights))

if __name__ == '__main__': unittest.main()
