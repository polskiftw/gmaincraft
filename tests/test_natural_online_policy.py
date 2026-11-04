"""Pure strict schema/learning tests, separate from live Minecraft evidence."""
import pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.natural_online_policy import (NaturalPPO,NaturalTransition,
     SCHEMA,SELF_FEATURES,MAX_OBJECTS,ID_BYTES,wood_id,observe,action_from_heads,
     PERSISTENCE_SCHEMA,PERSISTENCE_HEAD_SIZES,PERSISTENCE_DURATIONS,
     WOOD_ENGAGEMENT_SCHEMA,WOOD_ENGAGEMENT_SELF_FEATURES,
     MOTOR_V7_SCHEMA,MOTOR_V7,MOTOR_V7_HEAD_SIZES,
     action_from_motor_v7,motor_v7_history,motor_v7_duration_index,MOVE_KEYS,
     action_from_persistence_heads)
from gmaincraft.player_knowledge import PlayerKnowledgeV1,OwnedItem
from gmaincraft.mcio_status import PlayerStatusV1,CrosshairBlock
from gmaincraft.mcio_surface_objects import SurfaceObjectsV1,ProjectedBlockV1
from gmaincraft.mcio_mining_feedback import MiningFeedbackV1
from gmaincraft.natural_diagnostics import NaturalActionAudit,WoodCrackProgress


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

    def test_action_diagnostics_are_observational_only(self):
        audit=NaturalActionAudit()
        state=sample()
        hit=CrosshairBlock('minecraft:oak_log',(6,69,-1),'north',2.4)
        audit.observe((1,0,1),state,hit)
        audit.observe((2,0,1),state,hit)
        audit.observe((3,2,0),state,None)
        d=audit.summary()
        self.assertEqual(d['action_counts'],[[0,1,1,1,0,0],[2,0,1,0,0,0,0],[1,2]])
        self.assertEqual(d['attack_on_wood_decisions'],2)
        self.assertEqual(d['max_attack_still_streak'],2)
        self.assertEqual(d['crosshair_block_decisions'],2)
        with self.assertRaises(ValueError):audit.observe((1,2,7),state,None)

    def test_crack_progress_only_increments_once_on_same_observed_wood(self):
        from dataclasses import replace
        audit=WoodCrackProgress(max_blocks=1)
        obs=sample()
        oak=CrosshairBlock('minecraft:oak_log',(6,69,-1),'north',2.4)
        stone=CrosshairBlock('minecraft:stone',(6,69,-1),'north',2.4)
        def stage(t):
            own=obs.own.copy();own[22]=t/10
            return replace(obs,own=own)
        self.assertEqual(audit.observe(stage(1),oak),0.)  # first sample cannot establish progress
        self.assertAlmostEqual(audit.observe(stage(3),oak),.60)
        self.assertEqual(audit.observe(stage(2),oak),0.)
        self.assertAlmostEqual(audit.observe(stage(4),oak),.30)
        self.assertEqual(audit.observe(stage(1),stone),0.)
        self.assertEqual(audit.observe(stage(9),oak),0.)  # first frame after target change
        self.assertEqual(audit.observe(stage(10),oak),.30)
        other=CrosshairBlock('minecraft:birch_log',(7,69,-1),'north',2.4)
        self.assertEqual(audit.observe(stage(1),other),0.)
        self.assertEqual(audit.observe(stage(3),other),0.) # capped dedupe memory

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

    def test_learned_duration_actions_are_only_vanilla_key_holds(self):
        self.assertEqual(PERSISTENCE_DURATIONS,(6,24,72))
        self.assertEqual(PERSISTENCE_HEAD_SIZES,(6,7,2,3))
        action=action_from_persistence_heads((0,0,1,2))
        self.assertEqual((action.keys,action.look_dx,action.ticks),(('LEFT_BUTTON',),0.,72))
        self.assertEqual(action_from_persistence_heads((1,0,1,2),remaining_ticks=18).ticks,18)
        self.assertEqual(action_from_persistence_heads((1,3,0,1)).look_dx,-200)
        for invalid in ((1,0,1),(1,0,1,3),(False,0,1,2),(2,7,1,1)):
            with self.assertRaises(ValueError):action_from_persistence_heads(invalid)
        with self.assertRaises(ValueError):action_from_persistence_heads((0,0,1,2),remaining_ticks=0)

    def test_v4_model_trains_four_heads_without_loading_old_three_head_checkpoint(self):
        trainer=NaturalPPO(device='cpu',seed=705,schema=PERSISTENCE_SCHEMA,
                           action_sizes=PERSISTENCE_HEAD_SIZES,entropy_coef=.04)
        obs=sample()
        heads,logp,value=trainer.act(obs)
        self.assertEqual(len(heads),4)
        self.assertEqual(trainer.state()['action_sizes'],PERSISTENCE_HEAD_SIZES)
        traj=[NaturalTransition(obs,heads,.3,value,logp,72,False)]*4
        self.assertEqual(trainer.update(traj,bootstrap_value=value)['update'],1)
        other=NaturalPPO(device='cpu',seed=706,schema=PERSISTENCE_SCHEMA,
                         action_sizes=PERSISTENCE_HEAD_SIZES)
        other.restore(trainer.state())
        self.assertEqual(other.act(obs,greedy=True),trainer.act(obs,greedy=True))
        old=NaturalPPO(device='cpu',seed=705)
        with self.assertRaises(ValueError):old.restore(trainer.state())
        with self.assertRaises(ValueError):trainer.restore(old.state())
        with self.assertRaises(ValueError):NaturalPPO(device='cpu',schema=PERSISTENCE_SCHEMA)

    def test_v6_explicit_wood_features_are_from_same_visible_records(self):
        from dataclasses import replace
        oak=sample('minecraft:oak_log')
        # Reconstruct only from the SAME typed known player/status/camera.
        player=PlayerKnowledgeV1('player_knowledge_v1',(5.5,69.0,-3.5),0.,14.,20.,(),(),(),())
        status=PlayerStatusV1('gmaincraft_status_v1',20,5.,0,300,0,
                             'minecraft:overworld',
                             CrosshairBlock('minecraft:oak_log',(6,69,-1),'north',2.4))
        surface=SurfaceObjectsV1('gmaincraft_surface_objects_v1',13,9,70.,64.,(
            ProjectedBlockV1('minecraft:oak_log',(6,69,-1),('north',),2.5,15,0,(50,51),2/117,
                             (6/13,4/9),(.4,.45,.6,.55)),))
        v6=observe(player,status,surface,None,start_xyz=(5.5,69,-3.5),
                   schema=WOOD_ENGAGEMENT_SCHEMA)
        self.assertEqual(v6.own.shape,(len(WOOD_ENGAGEMENT_SELF_FEATURES),))
        self.assertEqual(tuple(v6.own[-2:]),(1.,1/24))
        missed=observe(player,replace(status,crosshair_block=None),surface,None,
                       start_xyz=(5.5,69,-3.5),schema=WOOD_ENGAGEMENT_SCHEMA)
        self.assertEqual(missed.own[-2],0.)
        self.assertEqual(missed.own[-1],1/24)
        legacy=observe(player,status,surface,None,start_xyz=(5.5,69,-3.5))
        self.assertEqual(legacy.own.shape,(len(SELF_FEATURES),))
        with self.assertRaises(ValueError):
            observe(player,status,surface,None,start_xyz=(5.5,69,-3.5),schema='guess')

    def test_v6_scratch_model_feature_schema_and_strict_roundtrip(self):
        trainer=NaturalPPO(device='cpu',seed=702,schema=WOOD_ENGAGEMENT_SCHEMA,
                           action_sizes=PERSISTENCE_HEAD_SIZES,entropy_coef=.06)
        # Shape must match V6, even if original v4/v5 numeric features are similar.
        state=sample()
        from dataclasses import replace
        v6=replace(state,own=np.concatenate([state.own,np.array([1.,1/24],np.float32)]))
        act,logp,value=trainer.act(v6)
        self.assertEqual(len(act),4)
        self.assertEqual(trainer.state()['feature_order'],WOOD_ENGAGEMENT_SELF_FEATURES)
        self.assertEqual(trainer.update([NaturalTransition(v6,act,.4,value,logp,24,False)]*4)['update'],1)
        saved=trainer.state()
        restored=NaturalPPO(device='cpu',seed=703,schema=WOOD_ENGAGEMENT_SCHEMA,
                            action_sizes=PERSISTENCE_HEAD_SIZES)
        restored.restore(saved)
        self.assertEqual(restored.act(v6,greedy=True),trainer.act(v6,greedy=True))
        prior=NaturalPPO(device='cpu',schema=PERSISTENCE_SCHEMA,
                         action_sizes=PERSISTENCE_HEAD_SIZES)
        with self.assertRaises(ValueError):prior.restore(saved)
        with self.assertRaises(ValueError):trainer.restore(prior.state())

    def test_v7_motor_primitives_replace_rare_factor_combinations_without_target_selector(self):
        self.assertEqual(MOTOR_V7_HEAD_SIZES,(20,))
        self.assertEqual(len(set(row[0] for row in MOTOR_V7)),len(MOTOR_V7))
        self.assertEqual(len(set(row[1:] for row in MOTOR_V7)),len(MOTOR_V7))
        self.assertEqual(set(x[6] for x in MOTOR_V7),{6,24,72})
        for idx, (label,move,look,attack,dx,dy,ticks) in enumerate(MOTOR_V7):
            action=action_from_motor_v7((idx,))
            self.assertEqual((action.look_dx,action.look_dy,action.ticks),(dx,dy,ticks))
            self.assertEqual(action.keys,MOVE_KEYS[move]+(('LEFT_BUTTON',) if attack else ()))
            self.assertEqual(motor_v7_history((idx,)),(move,look,attack))
            self.assertEqual(motor_v7_duration_index((idx,)),(6,24,72).index(ticks))
            self.assertNotIn('commands',vars(action))
        self.assertEqual(action_from_motor_v7((14,)).keys,('LEFT_BUTTON',))
        self.assertEqual(action_from_motor_v7((14,)).ticks,72)
        self.assertEqual(action_from_motor_v7((14,),remaining_ticks=18).ticks,18)
        self.assertEqual(action_from_motor_v7((6,)).look_dx,-24.)
        # No branch in the motor implementation can inspect the environment;
        # these motor choices are intentionally unconditional.
        for invalid in ((20,),(-1,),(True,),(),(0,1),(2.0,)):
            with self.assertRaises(ValueError):action_from_motor_v7(invalid)
        with self.assertRaises(ValueError):action_from_motor_v7((14,),remaining_ticks=0)

    def test_v7_new_policy_schema_and_scratch_ppo_update(self):
        import torch
        from dataclasses import replace
        obs=sample()
        v7=replace(obs,own=np.concatenate([obs.own,np.array([1.,1/24],np.float32)]))
        actor=NaturalPPO(device='cpu',seed=719,schema=MOTOR_V7_SCHEMA,
                         action_sizes=MOTOR_V7_HEAD_SIZES,entropy_coef=.04)
        heads,logp,value=actor.act(v7)
        self.assertEqual(len(heads),1)
        self.assertEqual(actor.state()['feature_order'],WOOD_ENGAGEMENT_SELF_FEATURES)
        self.assertEqual(actor.update([NaturalTransition(v7,heads,.25,value,logp,72,False)]*8)['update'],1)
        restored=NaturalPPO(device='cpu',seed=720,schema=MOTOR_V7_SCHEMA,
                            action_sizes=MOTOR_V7_HEAD_SIZES)
        restored.restore(actor.state())
        self.assertEqual(restored.act(v7,greedy=True),actor.act(v7,greedy=True))
        legacy=NaturalPPO(device='cpu',schema=WOOD_ENGAGEMENT_SCHEMA,
                          action_sizes=PERSISTENCE_HEAD_SIZES)
        with self.assertRaises(ValueError):legacy.restore(actor.state())
        with self.assertRaises(ValueError):actor.restore(legacy.state())

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
