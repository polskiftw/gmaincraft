import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from gmaincraft.natural_eval import make_agent, plan_runs


class NaturalEvalTests(unittest.TestCase):
    def test_distinct_seed_worlds_per_mode(self):
        plans = plan_runs('a01', [42, 7001], ['idle', 'legacy_probe'])
        self.assertEqual(len(plans), 4)
        self.assertEqual(len({p.world for p in plans}), 4)
        self.assertEqual(len({p.source for p in plans}), 2)
        self.assertEqual(plans[2].seed, 7001)
        self.assertEqual(plans[2].mode, 'idle')

    def test_invalid_plan_rejected(self):
        for args in [('../bad', [1], ['idle']), ('ok', [1, 1], ['random']),
                     ('ok', [1], ['bad']), ('ok', [2**65], ['idle']),
                     ('ok', [1], ['random', 'random']), ('ok', [], ['idle'])]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                plan_runs(*args)

    def test_basic_agents_only_return_normal_inputs(self):
        for name in ('idle', 'forward_attack', 'random'):
            agent = make_agent(name, 4)
            for budget in (1, 7, 42):
                action = agent.act(None, budget)
                self.assertLessEqual(action.ticks, budget)
                self.assertGreaterEqual(action.ticks, 1)
                self.assertFalse(hasattr(action, 'commands'))

    def test_legacy_requires_frozen_checkpoints(self):
        with self.assertRaises(ValueError):
            make_agent('legacy_probe', 5)


if __name__ == '__main__':
    unittest.main()

class WorldFixtureCheck(unittest.TestCase):
    def test_prepare_refuses_nonexistent_nbt(self):
        import tempfile
        from experiments.natural_world_benchmark import fix_spawn_and_verify
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                fix_spawn_and_verify(pathlib.Path(d), 42)
