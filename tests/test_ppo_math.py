"""NumPy-only semi-Markov GAE tests (also run in lightweight GitHub CI)."""
import pathlib
import sys
import unittest
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from gmaincraft.ppo_math import timed_gae


class TimedGAETest(unittest.TestCase):
    def test_terminal_reward_never_leaks_into_following_episode(self):
        a, target = timed_gae([1., 100.], [0., 0.], [5, 5], [True, True],
                              gamma_tick=1., lam_per_five_ticks=1.)
        np.testing.assert_allclose(a, [1., 100.])
        np.testing.assert_allclose(target, [1., 100.])

    def test_variable_duration_actually_changes_returns(self):
        a, _ = timed_gae([0., 1.], [0., 0.], [1, 1], [False, True],
                         gamma_tick=.9, lam_per_five_ticks=1.)
        b, _ = timed_gae([0., 1.], [0., 0.], [10, 1], [False, True],
                         gamma_tick=.9, lam_per_five_ticks=1.)
        self.assertAlmostEqual(a[0], .9, places=6)
        self.assertAlmostEqual(b[0], .9**10, places=6)
        self.assertLess(b[0], a[0])

    def test_nonterminal_bootstrap(self):
        advantage, target = timed_gae([0.], [2.], [3], [False],
                            last_value=5., gamma_tick=.9,
                            lam_per_five_ticks=.95)
        self.assertAlmostEqual(float(target[0]), 5*(.9**3), places=5)
        self.assertAlmostEqual(float(advantage[0]), 5*(.9**3)-2, places=5)

    def test_invalid_durations_and_shapes(self):
        for a, b, dt, done in (([1], [0], [0], [True]),
                                ([1, 2], [0], [3, 3], [True, True]),
                                ([1], [0], [-3], [False])):
            with self.assertRaises(ValueError):
                timed_gae(a,b,dt,done)


if __name__=='__main__': unittest.main()
