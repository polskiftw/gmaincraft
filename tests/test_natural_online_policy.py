"""Pure strict schema/learning tests, separate from live Minecraft evidence."""
import pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.natural_online_policy import (NaturalPPO,NaturalTransition,
     SCHEMA,SELF_FEATURES,MAX_OBJECTS,ID_BYTES,wood_id,observe,action_from_heads)
from gmaincraft.player_knowledge import PlayerKnowledgeV1,OwnedItem
from gmaincraft.mcio_status import PlayerStatusV1,CrosshairBlock
from gmaincraft.mcio_surface_objects import SurfaceObjectsV1,ProjectedBlockV1
from gmaincraft.mcio_mining_feedback import MiningFeedbackV1


def sample(wood='minecraft:oak_log'):
    player=PlayerKnowledgeV1('player_knowledge_v1',(5.5,69.0,-3.5),0.,14.,20.,
                             (OwnedItem(0,wood,1),),(),(),())
    target=CrosshairBlock(wood,(6,69,-1),'north',2.4)
    status=PlayerStatusV1('gmaincraft_status_v1',20,5.,0,300,0,
                          'minecraft:overworld',target)
    surface=SurfaceObjectsV1('gmaincraft_surface_objects_v1',13,9,70.,64.,(
        ProjectedBlockV1(wood,(6,69,-1),('north',),2.5,15,0,(50,51),2/117,
                         (6/13,4/9),(.4,.45,.6,.55)),))
    mining=MiningFeedbackV1('gmaincraft_mining_feedback_v1',True,5,True)
    return observe(player,status,surface,mining,start_xyz=(5.5,69,-3.5),
                   last_action=(1,2,1),visit_count=3,elapsed_ticks=120)


class NaturalPolicyTests(unittest.TestCase):
    def test_typed_observation_does_not_encode_world_or_cheat(self):
        state=sample()
        self.assertEqual(state.own.shape,(len(SELF_FEATURES),))
        self.assertEqual(state.crosshair_id.shape,(ID_BYTES,))
        self.assertEqual(state.surface_ids.shape,(MAX_OBJECTS,ID_BYTES))
        self.assertEqual(state.surface_mask[0],1.)
        self.assertEqual(state.surface_mask[1],0.)
        self.assertTrue(np.isfinite(state.own).all())
        self.assertFalse(state.own.flags.writeable)
        with self.assertRaises(ValueError):state.own[0]=2
        self.assertNotIn('seed',','.join(SELF_FEATURES))
        self.assertNotIn('target_location',','.join(SELF_FEATURES))
        self.assertTrue(wood_id('minecraft:cherry_log'))
        self.assertFalse(wood_id('minecraft:stone'))

    def test_only_vanilla_motor_choices(self):
        action=action_from_heads((1,2,1))
        self.assertEqual(action.keys,('W','LEFT_BUTTON'))
        self.assertEqual(action.ticks,6)
        self.assertNotIn('commands',vars(action))
        for invalid in ((6,0,0),(0,7,0),(0,0,2),(0,False,0)):
            with self.assertRaises(ValueError):action_from_heads(invalid)

    def test_truly_scratch_ppo_update_and_safe_checkpoint_state(self):
        trainer=NaturalPPO(device='cpu',seed=700)
        obs=sample()
        act,logp,value=trainer.act(obs)
        self.assertEqual(len(act),3)
        self.assertTrue(np.isfinite(logp) and np.isfinite(value))
        traj=[NaturalTransition(obs,act,.2,value,logp,6,False)]*4
        metrics=trainer.update(traj,bootstrap_value=value)
        self.assertEqual(metrics['update'],1)
        checkpoint=trainer.state()
        self.assertEqual(checkpoint['schema'],SCHEMA)
        other=NaturalPPO(device='cpu',seed=701)
        other.restore(checkpoint)
        self.assertEqual(other.updates,1)
        self.assertEqual(other.act(obs,greedy=True),trainer.act(obs,greedy=True))
        bad=dict(checkpoint,schema='other')
        with self.assertRaises(ValueError):other.restore(bad)

    def test_cuda_checkpoint_roundtrip_device_safe(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('NVIDIA CUDA not available')
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path=pathlib.Path(td)/'checkpoint.pt'
            original=NaturalPPO(device='cuda',seed=702)
            obs=sample()
            before=original.act(obs,greedy=True)
            torch.save(original.state(),path)
            saved=torch.load(path,map_location='cuda',weights_only=True)
            restored=NaturalPPO(device='cuda',seed=703)
            restored.restore(saved)
            self.assertEqual(restored.act(obs,greedy=True),before)

if __name__=='__main__':unittest.main()
