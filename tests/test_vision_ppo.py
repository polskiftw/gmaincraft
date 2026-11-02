"""Lightweight vision/actor-critic smoke (skipped when optional torch absent)."""
import importlib.util
import pathlib
import sys
import unittest
import numpy as np
from types import SimpleNamespace
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

HAS_TORCH = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('PIL') is not None


@unittest.skipUnless(HAS_TORCH,'Optional GPU/PyTorch deps are not installed in CI')
class VisionPPOTest(unittest.TestCase):
    def test_pixel_history_and_macros(self):
        from gmaincraft.vision_ppo import PixelHistory, action_for_index, ACTION_COUNT
        view=SimpleNamespace(pixels=np.zeros((120,240,3),dtype=np.uint8))
        h=PixelHistory(); state=h.reset(view)
        self.assertEqual(state.shape,(6,84,84))
        self.assertEqual(state.dtype, np.uint8)
        view.pixels[:,:,:]=90
        state=h.append(view)
        self.assertEqual(int(state[0,0,0]),0)
        self.assertEqual(int(state[3,0,0]),90)
        self.assertEqual(action_for_index(8).ticks,80)
        self.assertEqual(action_for_index(8,7).ticks,7)
        for i in range(ACTION_COUNT):
            self.assertFalse(hasattr(action_for_index(i),'commands'))

    def test_cpu_forward_update_checkpoint_and_cuda_strictness(self):
        import tempfile
        import torch
        from gmaincraft.vision_ppo import PPOTrainer, Transition
        p=PPOTrainer(device='cpu',seed=86,batch_size=4,update_epochs=1)
        self.assertGreater(p.parameter_count, 50000)
        state=np.zeros((6,84,84),dtype=np.uint8)
        a,logp,value=p.act(state)
        self.assertTrue(0<=a<11)
        self.assertTrue(np.isfinite(logp) and np.isfinite(value))
        before=p.net.actor.weight.detach().clone()
        traj=[Transition(np.full_like(state,i*10),a,1.,value,logp,5,i==3)
              for i in range(4)]
        result=p.update(traj)
        self.assertEqual(result['device'],'cpu')
        self.assertTrue(np.isfinite(result['loss']))
        self.assertFalse(torch.equal(before,p.net.actor.weight))
        with tempfile.TemporaryDirectory() as d:
            path=pathlib.Path(d)/'test.pt';p.save(path,{'seed':86})
            restored=PPOTrainer(device='cpu',seed=99)
            self.assertEqual(restored.load_weights(path),{'seed':86})
            self.assertTrue(torch.equal(restored.net.actor.weight,p.net.actor.weight))
            self.assertEqual(restored.load_weights(path,restore_optimizer=True),{'seed':86})
            self.assertGreater(len(restored.optim.state),0)
            self.assertEqual(list(path.parent.glob('*.tmp')),[])
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, 'CUDA requested'):
                PPOTrainer(device='cuda')


if __name__=='__main__': unittest.main()
