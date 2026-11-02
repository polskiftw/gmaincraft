import pathlib
import sys
import unittest
import math
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.joint_curriculum import sample_near_cases,sample_stationary_cases


class NearCaseTests(unittest.TestCase):
    def test_reproducible_near_scenarios_distinct_from_broad(self):
        a=sample_near_cases(1001,100)
        self.assertEqual(a,sample_near_cases(1001,100))
        self.assertNotEqual(a,sample_near_cases(1002,100))
        for case in a:
            d=math.hypot(case.tree_dx-case.player_dx,case.tree_dz-case.player_dz)
            self.assertLessEqual(d,3.7)
            self.assertGreaterEqual(d,1.8)
            bearing=math.degrees(math.atan2(-(case.tree_dx-case.player_dx),case.tree_dz-case.player_dz))
            self.assertLessEqual(abs(case.yaw-bearing),24)
        self.assertGreater(len({(c.tree_dx,c.tree_dz) for c in a}),4)

    def test_stationary_aimed_fixture(self):
        cases=sample_stationary_cases(177,35)
        self.assertEqual(cases,sample_stationary_cases(177,35))
        self.assertTrue(all(c.tree_dx==0 and c.tree_dz==2 and c.player_dx==0 and c.player_dz==0
                            and abs(c.yaw)<=4.5 and abs(c.pitch-29.25)<=4.5 for c in cases))

    def test_invalid_count(self):
        with self.assertRaises(ValueError):sample_near_cases(1,0)
        with self.assertRaises(ValueError):sample_stationary_cases(1,0)

if __name__=='__main__':unittest.main()


class MixedCurriculumTests(unittest.TestCase):
    def test_interleaving_preserves_cases_balances_stages(self):
        from collections import Counter
        from gmaincraft.joint_curriculum import interleave_phases
        rows=([('stationary',i) for i in range(20)]+
              [('aim',i) for i in range(30)]+[('near',i) for i in range(50)])
        ordered=interleave_phases(rows)
        self.assertEqual(Counter(rows),Counter(ordered))
        self.assertEqual(len(ordered),100)
        for phase in ('stationary','aim','near'):
            self.assertEqual([r[1] for r in ordered if r[0]==phase],
                             [r[1] for r in rows if r[0]==phase])
        first=Counter(row[0] for row in ordered[:10])
        self.assertGreater(first['stationary'],0)
        self.assertGreater(first['aim'],0)
        self.assertGreater(first['near'],0)
        self.assertEqual(ordered,interleave_phases(rows))
        self.assertEqual(interleave_phases([]),[])
