"""Pure feature/motor contract for from-scratch attack skill; NOT game proof."""
import pathlib
import sys
import tempfile
import unittest

import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.mcio_status import PlayerStatusV1,CrosshairBlock
from gmaincraft.mcio_mining_feedback import MiningFeedbackV1,SCHEMA as FEEDBACK_SCHEMA
from gmaincraft.mining_attack_policy import (MiningAttackLearner,AttackStep,
        SCHEMA,FEATURE_NAMES,FEATURE_SIZE,encode_attack_observation,
        action_for_attack,step_reward,discounted_returns)


class AttackMotorTests(unittest.TestCase):
    def context(self):
        status=PlayerStatusV1('gmaincraft_status_v1',20,5.,0,300,0,
                              'minecraft:overworld',CrosshairBlock(
                                   'minecraft:oak_log',(4,66,9),'north',2.3))
        feedback=MiningFeedbackV1(FEEDBACK_SCHEMA,True,5,True)
        return status,feedback

    def test_state_is_bounded_and_readonly(self):
        a,b=self.context()
        values=encode_attack_observation(a,b,previous_attack=True,elapsed_ticks=30)
        self.assertEqual(values.shape,(FEATURE_SIZE,))
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values>=0).all() and (values<=1).all())
        self.assertEqual(FEATURE_NAMES[3],'crack_stage_per_10')
        self.assertEqual(values[3],.5)
        self.assertEqual(values[4],1)
        with self.assertRaises(ValueError):values[0]=0
        self.assertNotIn('tree',','.join(FEATURE_NAMES))
        for a,b in ((None,None),(1,10)):
            with self.assertRaises((TypeError,ValueError)):
                encode_attack_observation(a,b,previous_attack=True,elapsed_ticks=1)

    def test_allowed_actions_never_administrative(self):
        self.assertEqual(action_for_attack(1).keys,('LEFT_BUTTON',))
        self.assertEqual(action_for_attack(0,4).ticks,4)
        self.assertNotIn('commands',vars(action_for_attack(1)))
        for invalid in (-1,2,False,1.0):
            with self.assertRaises(ValueError):action_for_attack(invalid)

    def test_crack_progress_and_release_shaping(self):
        _,b=self.context()
        future=MiningFeedbackV1(FEEDBACK_SCHEMA,True,6,True)
        released=MiningFeedbackV1(FEEDBACK_SCHEMA,False,-1,True)
        self.assertGreater(step_reward(b,future,mined_new_block=False),0.)
        self.assertLess(step_reward(b,released,mined_new_block=False),0.)
        self.assertGreater(step_reward(b,released,mined_new_block=True),10.)
        steps=[AttackStep(np.zeros(FEATURE_SIZE,dtype=np.float32),1,x)
               for x in (.2,.3,.8)]
        self.assertEqual(len(discounted_returns(steps)),3)
        with self.assertRaises(ValueError):discounted_returns([])

    def test_tiny_cuda_or_cpu_neural_update_checkpoint_schema(self):
        a,b=self.context()
        state=encode_attack_observation(a,b,previous_attack=True,elapsed_ticks=10)
        model=MiningAttackLearner(seed=4122,device='cpu')
        choice,p=model.act(state)
        self.assertIn(choice,(0,1))
        self.assertTrue(0<=p<=1)
        result=model.update([AttackStep(state,choice,.3),
                             AttackStep(state,choice,1.5)])
        self.assertEqual(result['transitions'],2)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            dest=pathlib.Path(tmp)/'test-motor.pt'
            model.save(dest,dict(schema=SCHEMA,init='random_weights',train_seed=123))
            with self.assertRaises(FileExistsError):model.save(dest,{})
            clone=MiningAttackLearner(seed=99,device='cpu')
            metadata=clone.load(dest)
            self.assertEqual(metadata['train_seed'],123)
            self.assertEqual(clone.updates,1)
            self.assertEqual(clone.act(state,greedy=True),model.act(state,greedy=True))

if __name__=='__main__':unittest.main()
