"""Factorized input contract; optional Torch test skipped in minimal CI."""
import importlib.util
import pathlib
import sys
import unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.vision_factorized import ACTION_TICKS,HEAD_SIZES,SCHEMA,action_for_factors


class FactorActionTests(unittest.TestCase):
    def test_simultaneous_movement_look_attack(self):
        a=action_for_factors((1,2,1))
        self.assertEqual(a.keys,('W','LEFT_BUTTON'))
        self.assertEqual(a.look_dx,60.)
        self.assertEqual(a.ticks,6)
        self.assertEqual(action_for_factors((5,4,1),2).keys,('W','SPACE','LEFT_BUTTON'))
        self.assertEqual(action_for_factors((5,4,1),2).ticks,2)
        self.assertEqual(HEAD_SIZES,(6,5,2))
        self.assertEqual(ACTION_TICKS,6)

    def test_invalid_or_privileged_inputs(self):
        for value in [(6,0,0),(0,5,0),(0,0,2),(-1,0,0),(0,),(True,0,0),(0,'give',1)]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                action_for_factors(value)
        with self.assertRaises(ValueError): action_for_factors((0,0,0),0)
        self.assertNotIn('commands', vars(action_for_factors((1,1,1))))


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'torch optional')
class TorchFactorTests(unittest.TestCase):
    def test_ppo_gradient_cuda_optional(self):
        from gmaincraft.vision_factorized import FactorizedPPOTrainer,FactorTransition
        import torch
        t=FactorizedPPOTrainer(device='cpu',seed=77,batch_size=4,update_epochs=1)
        image=np.zeros((6,84,84),dtype=np.uint8)
        a,lp,v=t.act(image)
        self.assertEqual(len(a),3)
        masked,masked_lp,_=t.act(image,active_heads=(False,False,True))
        self.assertEqual(masked[:2],(0,0))
        self.assertTrue(np.isfinite(masked_lp))
        with self.assertRaises(ValueError):t.act(image,active_heads=(False,False,False))
        self.assertTrue(np.isfinite(lp) and np.isfinite(v))
        before=t.net.camera.weight.detach().clone()
        traj=[FactorTransition(np.full_like(image,i*9),a,1. if i==3 else 0.,v,lp,6,i==3) for i in range(4)]
        metrics=t.update(traj)
        self.assertEqual(metrics['device'],'cpu')
        self.assertFalse(torch.equal(before,t.net.camera.weight))
        self.assertTrue(np.isfinite(metrics['loss']))
        # The same PPO objective must support an attack-only stage without
        # attributing log probability to disabled move/look channels.
        m,mlp,mv=t.act(image,active_heads=(False,False,True))
        masked_traj=[FactorTransition(np.full_like(image,i*7),m,.2,mv,mlp,6,
                                      i==3,(False,False,True)) for i in range(4)]
        mask_metrics=t.update(masked_traj)
        self.assertTrue(np.isfinite(mask_metrics['loss']))
        with __import__('tempfile').TemporaryDirectory() as d:
            from pathlib import Path
            path=Path(d)/'v2.pt'
            t.save(path,dict(schema=SCHEMA))
            copy=FactorizedPPOTrainer(device='cpu')
            self.assertEqual(copy.load_weights(path,restore_optimizer=True),dict(schema=SCHEMA))
            self.assertTrue(torch.equal(t.net.camera.weight,copy.net.camera.weight))

if __name__=='__main__':unittest.main()
