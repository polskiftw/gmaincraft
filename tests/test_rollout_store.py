import json
import pathlib
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.rollout_store import (arrays_from_trajectory,save_rollout_batch,
                                     read_rollout_batch,ALLOWED_FIELDS)
from gmaincraft.vision_factorized import FactorTransition,SCHEMA


class RolloutTest(unittest.TestCase):
    def test_roundtrip_does_not_store_privileged_geometry(self):
        image=np.full((6,84,84),77,dtype=np.uint8)
        steps=[FactorTransition(image.copy(),(0,1,1),.12,2.,-1.4,6,False,
                                (False,True,True)),
               FactorTransition(image.copy(),(1,0,1),20.,4.,-2.1,6,True,
                                (True,True,True))]
        meta=dict(schema=SCHEMA,run_tag='unit',end_episode=2,episodes=[1,2],
                  stages=['aim','near'],device='cpu',training_split='train')
        data=arrays_from_trajectory(steps,meta)
        self.assertEqual(set(data),ALLOWED_FIELDS)
        with tempfile.TemporaryDirectory() as d:
            p=save_rollout_batch(d,2,steps,meta)
            self.assertTrue(p.exists())
            loaded=read_rollout_batch(p)
            self.assertEqual(loaded['metadata'],meta)
            np.testing.assert_array_equal(loaded['rgb'],np.stack([image,image]))
            np.testing.assert_array_equal(loaded['action'],[[0,1,1],[1,0,1]])
            np.testing.assert_array_equal(loaded['done'],[False,True])
            self.assertNotIn('coordinates',loaded)
            self.assertNotIn('target',loaded)
            with self.assertRaises(FileExistsError):save_rollout_batch(d,2,steps,meta)

    def test_validation(self):
        with self.assertRaises(ValueError):arrays_from_trajectory([],{})
        t=FactorTransition(np.zeros((6,84,84),dtype=np.uint8),(0,1,1),.1,0.,-.4,6,True)
        with self.assertRaises(ValueError):arrays_from_trajectory([t],{'target_pos':[1,2,3]})
        t.action=(0,5,0)
        with self.assertRaises(ValueError):arrays_from_trajectory([t],{})

if __name__=='__main__':unittest.main()
