"""Pure data-label alignment tests: no Minecraft runtime needed."""
import pathlib
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.inverse_dynamics import load_rgb_action_pairs,INVERSE_SCHEMA
from gmaincraft.vision_factorized import FactorTransition,SCHEMA
from gmaincraft.rollout_store import save_rollout_batch


class InverseTest(unittest.TestCase):
    def test_action_causes_next_frame_not_previous_or_reset(self):
        with tempfile.TemporaryDirectory() as d:
            frames=[]
            for episode in (1,2):
                steps=[]
                for i in range(4):
                    image=np.zeros((6,84,84),dtype=np.uint8)
                    image[:3,:,:]=episode*40+i+1
                    image[3:,:,:]=episode*40+i+2
                    steps.append(FactorTransition(image,(i%6,i%5,i%2),0.,0.,-.9,6,
                                                  i==3,(True,True,True)))
                save_rollout_batch(d,episode,steps,dict(schema=SCHEMA,run_tag='unit',
                                                        end_episode=episode))
            x,y,episode_tags=load_rgb_action_pairs(d)
            self.assertEqual(x.shape,(6,6,84,84))
            self.assertEqual(y.shape,(6,3))
            self.assertEqual(len(set(episode_tags)),2)
            # Pre frame at decision i uses image[3:], post frame is next
            # decision image[3:]. We must NEVER pair last frame of one
            # episode with first frame from next (automatic reset).
            np.testing.assert_array_equal(x[0,:,0,0],[42,42,42,43,43,43])
            np.testing.assert_array_equal(x[2,:,0,0],[44,44,44,45,45,45])
            np.testing.assert_array_equal(x[3,:,0,0],[82,82,82,83,83,83])
            self.assertEqual(y[0].tolist(),[0,0,0])
            self.assertEqual(y[2].tolist(),[2,2,0])

    def test_no_files(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):load_rgb_action_pairs(d)

if __name__=='__main__':unittest.main()
