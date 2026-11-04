"""Pure sensitivity metrics; no claim that artificial history ablation is play."""
import pathlib
import sys
import unittest
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments'))
from visible_target_policy_probe import summarize_metrics,WOOD_FIELD,CONTINUITY_FIELD
from gmaincraft.natural_online_policy import TEMPORAL_V9_SELF_FEATURES,MOTOR_V7


class SensitivityMetricsTests(unittest.TestCase):
    def test_wood_and_continuity_count_only_actual_sensor_fields(self):
        x=np.full((3,len(MOTOR_V7)),1/len(MOTOR_V7),np.float32)
        y=x.copy();y[1,14]+=0.05;y[1,0]-=0.05
        own=np.zeros((3,len(TEMPORAL_V9_SELF_FEATURES)),np.float32)
        own[1,WOOD_FIELD]=1.
        own[2,CONTINUITY_FIELD]=1.
        result=summarize_metrics(y,x,own)
        self.assertEqual(result['observed_decisions'],3)
        self.assertEqual(result['actual_wood_crosshair_decisions'],1)
        self.assertEqual(result['actual_previous_same_target_decisions'],1)
        self.assertGreater(result['mean_total_variation_from_history_ablation'],0.)
        self.assertAlmostEqual(result['mean_stationary_attack_probability_on_actual_wood'],.2)
        with self.assertRaises(ValueError):summarize_metrics(y[:1],x,own)


if __name__=='__main__':unittest.main()
