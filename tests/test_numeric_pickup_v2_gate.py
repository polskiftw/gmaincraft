"""Never promote a learner from raw rewards, mismatched data or one seed."""
import pathlib,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import importlib.util
spec=importlib.util.spec_from_file_location('pickup_gate',ROOT/'tools/gate-numeric-pickup-v2.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class PickupPromotionGateTest(unittest.TestCase):
    def test_two_real_independent_case_suites(self):
        pairs=[('egoeval_greedy_1105a','erand1105a','efwd2_1105a'),
               ('egreed2_1105a','erand2_1105a','efwd3_1105a')]
        # Log-based regression: real experiments, not synthetic success labels.
        if all((ROOT/'logs'/('numeric_pickup_'+t+'.jsonl')).is_file()
               for triple in pairs for t in triple):
            self.assertTrue(mod.gate(pairs))
            with self.assertRaises(ValueError):mod.gate(pairs[:1])
            with self.assertRaises(ValueError):mod.gate([pairs[0],pairs[0]])
            self.assertFalse(mod.gate(pairs,threshold=.30))

if __name__=='__main__':unittest.main()
