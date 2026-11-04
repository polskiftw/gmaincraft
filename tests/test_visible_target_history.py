"""V9 pure tests: real-player history, no new motor/secret reward pathway."""
from dataclasses import replace
import pathlib
import sys
import tempfile
import unittest

import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments'))
from gmaincraft.agent_interface import AgentAction
from gmaincraft.mcio_mining_feedback import MiningFeedbackV1
from gmaincraft.mcio_status import CrosshairBlock,PlayerStatusV1
from gmaincraft.mcio_surface_objects import SurfaceObjectsV1
from gmaincraft.player_knowledge import PlayerKnowledgeV1
from gmaincraft.natural_online_policy import (
    NaturalPPO,NaturalTransition,MOTOR_V7_SCHEMA,MOTOR_V7_HEAD_SIZES,
    TEMPORAL_V9_SCHEMA,TEMPORAL_V9_FEATURES,TEMPORAL_V9_SELF_FEATURES,
    WOOD_ENGAGEMENT_SELF_FEATURES,observe)
from gmaincraft.visible_target_history import VisibleTargetHistory,FEATURE_NAMES,SCHEMA
from gmaincraft.experience_archive import EpisodeExperience,read_episode
from fresh_seed_treadmill import (parse_args,metadata,policy_schema,save_model,restore,
                                   event_reward_profile)


def status(block='minecraft:oak_log',xyz=(2,64,1)):
    hit=CrosshairBlock(block,xyz,'north',2.2) if block is not None else None
    return PlayerStatusV1('gmaincraft_status_v1',20,5.,0,300,0,'minecraft:overworld',hit)


def mining(stage,active=True):
    return MiningFeedbackV1('gmaincraft_mining_feedback_v1',active,stage,True)


def attack(ticks=6,move=False):
    return AgentAction(keys=(('W',) if move else ())+('LEFT_BUTTON',),ticks=ticks)


def state(history):
    player=PlayerKnowledgeV1('player_knowledge_v1',(0.5,64.,0.5),0.,0.,20.,(),(),(),())
    surfaces=SurfaceObjectsV1('gmaincraft_surface_objects_v1',13,9,70.,64.,())
    return observe(player,status(),surfaces,mining(3),start_xyz=player.player_xyz,
                   schema=TEMPORAL_V9_SCHEMA,target_history=history)


class TemporalHistoryTests(unittest.TestCase):
    def test_bounded_own_target_and_real_crack_continuity(self):
        self.assertEqual(TEMPORAL_V9_FEATURES,FEATURE_NAMES)
        h=VisibleTargetHistory()
        first=h.observe(status(),mining(2),elapsed_ticks=0)
        self.assertEqual(len(first),10)
        self.assertEqual(first[0],0.)
        self.assertEqual(first[6],1.)
        self.assertEqual(first[8],.2)
        self.assertEqual(first[7],0.)
        second=h.observe(status(),mining(5),previous_action=attack(6),elapsed_ticks=6)
        self.assertEqual(second[0],1.)
        self.assertEqual(second[1],6/72)
        self.assertEqual(second[2],1.)
        self.assertEqual(second[3:5],(1.,1.))
        self.assertEqual(second[5],6/72)
        self.assertAlmostEqual(second[7],.3)
        self.assertEqual(second[8],.5)
        self.assertEqual(h.summary()['visible_crack_rise_samples'],1)
        third=h.observe(status(),mining(5),previous_action=attack(24),elapsed_ticks=24)
        self.assertAlmostEqual(third[5],30/72)
        self.assertEqual(third[9],24/72)
        fourth=h.observe(status(),mining(-1),previous_action=attack(6),elapsed_ticks=6)
        self.assertEqual(fourth[6],0.)
        self.assertEqual(fourth[7],0.)
        self.assertEqual(fourth[9],0.)
        self.assertEqual(fourth[8],.5)  # remembered actual previous crack high water
        self.assertTrue(all(0<=v<=1 for v in fourth))

    def test_miss_switch_movement_and_new_world_never_inherit_hidden_state(self):
        h=VisibleTargetHistory()
        h.observe(status(),mining(6),elapsed_ticks=0)
        missed=h.observe(status(block=None),None,previous_action=attack(6),elapsed_ticks=6)
        self.assertEqual(missed[0],0.)
        self.assertEqual(missed[2],1.)  # remembers old observed wood
        self.assertEqual(missed[6:10],(0.,0.,0.,0.))
        switched=h.observe(status('minecraft:stone'),mining(9),
                           previous_action=attack(6,move=True),elapsed_ticks=6)
        self.assertEqual(switched[0],0.)
        self.assertEqual(switched[4],0.)
        self.assertEqual(switched[5],0.)
        self.assertEqual(switched[7],0.)
        self.assertEqual(switched[8],.9)
        new=VisibleTargetHistory().observe(status(),mining(6),elapsed_ticks=0)
        self.assertEqual(new[0],0.)
        self.assertEqual(new[8],.6)
        self.assertEqual(new[2],0.)

    def test_history_refuses_fake_or_stale_inputs(self):
        h=VisibleTargetHistory()
        with self.assertRaises(TypeError):h.observe('raw world',None,elapsed_ticks=0)
        with self.assertRaises(TypeError):h.observe(status(),'stage',elapsed_ticks=0)
        with self.assertRaises(ValueError):h.observe(status(),mining(1),elapsed_ticks=6)
        with self.assertRaises(ValueError):h.observe(status(),mining(1),
                            elapsed_ticks=12,previous_action=attack(6))
        with self.assertRaises(ValueError):h.observe(status(),mining(1),
                            elapsed_ticks=0,previous_action=attack(6))

    def test_v9_observation_and_checkpoint_do_not_corrupt_v8(self):
        h=VisibleTargetHistory().observe(status(),mining(3),elapsed_ticks=0)
        obs=state(h)
        self.assertEqual(obs.own.shape,(len(TEMPORAL_V9_SELF_FEATURES),))
        self.assertFalse(obs.own.flags.writeable)
        self.assertEqual(tuple(obs.own[-10:]),h)
        with self.assertRaises(ValueError):state((0.,)*9)
        with self.assertRaises(ValueError):state((0.,)*9+(2.,))
        player=PlayerKnowledgeV1('player_knowledge_v1',(0.5,64.,0.5),0.,0.,20.,(),(),(),())
        surfaces=SurfaceObjectsV1('gmaincraft_surface_objects_v1',13,9,70.,64.,())
        with self.assertRaises(ValueError):observe(player,status(),surfaces,mining(3),
                             start_xyz=player.player_xyz,schema=MOTOR_V7_SCHEMA,target_history=h)
        train=NaturalPPO(device='cpu',seed=870,schema=TEMPORAL_V9_SCHEMA,
                         action_sizes=MOTOR_V7_HEAD_SIZES)
        act,logp,value=train.act(obs)
        self.assertEqual(len(act),1)
        self.assertEqual(train.update([NaturalTransition(obs,act,.1,value,logp,6,False)]*8)['update'],1)
        copy=NaturalPPO(device='cpu',seed=871,schema=TEMPORAL_V9_SCHEMA,
                        action_sizes=MOTOR_V7_HEAD_SIZES)
        copy.restore(train.state())
        self.assertEqual(copy.act(obs,greedy=True),train.act(obs,greedy=True))
        old=NaturalPPO(device='cpu',seed=870,schema=MOTOR_V7_SCHEMA,
                       action_sizes=MOTOR_V7_HEAD_SIZES)
        with self.assertRaises(ValueError):old.restore(train.state())
        with self.assertRaises(ValueError):train.restore(old.state())

    def test_v9_reward_is_v8_and_motor_same_but_metadata_separate(self):
        old=parse_args(['--run-tag','preservedv8','--reward-profile','wood_sustain_v8'])
        new=parse_args(['--run-tag','firstv9','--reward-profile','wood_temporal_v9'])
        self.assertEqual(policy_schema(old),MOTOR_V7_SCHEMA)
        self.assertEqual(policy_schema(new),TEMPORAL_V9_SCHEMA)
        self.assertEqual(metadata(old)['motor_v7_actions'],metadata(new)['motor_v7_actions'])
        self.assertNotIn('visible_target_history_features',metadata(old))
        self.assertEqual(metadata(new)['visible_target_history_features'],FEATURE_NAMES)
        self.assertEqual(metadata(new)['visible_target_history_schema'],SCHEMA)
        self.assertEqual(event_reward_profile('wood_temporal_v9'),'wood_sustain_v8')
        self.assertEqual(event_reward_profile('wood_sustain_v8'),'wood_sustain_v8')
        self.assertEqual(event_reward_profile('wood_motor_v7'),'wood_target_v5')
        with tempfile.TemporaryDirectory() as td:
            path=pathlib.Path(td)/'v9.pt'
            model=NaturalPPO(device='cpu',schema=TEMPORAL_V9_SCHEMA,
                             action_sizes=MOTOR_V7_HEAD_SIZES)
            save_model(path,model,metadata(new),1,12,{})
            self.assertEqual(restore(path,model,metadata(new))[1],12)
            with self.assertRaises(ValueError):restore(path,model,metadata(old))

    def test_v9_replay_can_roundtrip_new_schema(self):
        x=state(VisibleTargetHistory().observe(status(),mining(2),elapsed_ticks=0))
        player=PlayerKnowledgeV1('player_knowledge_v1',(0.5,64.,0.5),0.,0.,20.,(),(),(),())
        recorder=EpisodeExperience(observation_schema=TEMPORAL_V9_SCHEMA,action_sizes=(20,))
        recorder.add(x,(14,),x,player,player,duration=72,reward=.25,done=False)
        with tempfile.TemporaryDirectory() as td:
            recorder.save(td,'v9archive',0)
            meta,archive=read_episode(pathlib.Path(td)/'v9archive'/'episode_000000000000.npz')
            self.assertEqual(meta['observation_schema'],TEMPORAL_V9_SCHEMA)
            self.assertEqual(archive['before_own'].shape,(1,len(TEMPORAL_V9_SELF_FEATURES)))


if __name__=='__main__':unittest.main()
