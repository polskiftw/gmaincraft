import pathlib
import sys
import unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.recurrent_inverse import previous_action_onehots


class CausalHistoryTest(unittest.TestCase):
    def test_history_shift_never_includes_current_action(self):
        y=np.array([[1,2,1],[3,4,0],[5,1,1]],dtype=np.int64)
        x=previous_action_onehots(y)
        self.assertEqual(x.shape,(3,13))
        self.assertTrue((x[0]==0).all())
        self.assertEqual(x[1,:6].argmax(),1)
        self.assertEqual(x[1,6:11].argmax(),2)
        self.assertEqual(x[1,11:].argmax(),1)
        self.assertEqual(x[2,:6].argmax(),3)
        self.assertNotEqual(x[2,:6].argmax(),5)
    def test_invalid_indices(self):
        with self.assertRaises(ValueError):previous_action_onehots(np.array([[6,0,0]]))
        with self.assertRaises(ValueError):previous_action_onehots(np.zeros((1,4),dtype=int))

if __name__=='__main__':unittest.main()
