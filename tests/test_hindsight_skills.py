"""Goal training is learned from self-recorded actual outcomes, not recipes."""
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments'))
from gmaincraft.experience_archive import EpisodeExperience,read_episode,hindsight_goal_transitions,compact_retention
from gmaincraft.hindsight_skills import HindsightSkillModel,SCHEMA
from gmaincraft.natural_online_policy import NaturalObservation,MOTOR_V7_SCHEMA
from gmaincraft.player_knowledge import PlayerKnowledgeV1,OwnedItem
import train_hindsight_goals as trainer


def obs():
    return NaturalObservation(
        np.zeros(31,np.float32),np.zeros(48,np.uint8),np.zeros((24,48),np.uint8),
        np.zeros((24,10),np.float32),np.zeros(24,np.float32))


def player(items=()):
    return PlayerKnowledgeV1('player_knowledge_v1',(0.,65.,0.),0.,0.,20.,
                             tuple(OwnedItem(i,item,num) for i,(item,num) in enumerate(items)),
                             (),(),())


def example():
    state=obs()
    return dict(**{'before_'+field:getattr(state,field) for field in
               ('own','crosshair_id','surface_ids','surface_data','surface_mask')},
               goal_item_id='minecraft:oak_log',ticks_to_achievement=72,action=(14,))


class HindsightTests(unittest.TestCase):
    def test_scratch_model_can_train_on_owned_goal_and_restore(self):
        m=HindsightSkillModel(own_size=31,action_sizes=(20,),device='cpu',seed=71)
        ex=example()
        start=m.train_examples([ex]*3,batch_size=3)
        self.assertTrue(np.isfinite(start['mean_weighted_loss']))
        for i in range(12):m.train_examples([ex]*3,batch_size=3,seed=i)
        self.assertEqual(m.act(obs(),'minecraft:oak_log'),(14,))
        saved=m.state()
        new=HindsightSkillModel(own_size=31,action_sizes=(20,),device='cpu',seed=72)
        new.restore(saved)
        self.assertEqual(new.act(obs(),'minecraft:oak_log'),(14,))
        self.assertEqual(saved['schema'],SCHEMA)
        with self.assertRaises(ValueError):
            HindsightSkillModel(own_size=29,action_sizes=(20,),device='cpu').restore(saved)
        with self.assertRaises(TypeError):m.act('raw world','minecraft:oak_log')
        with self.assertRaises(ValueError):m.act(obs(),'unseen')

    def test_not_credit_goal_that_prior_state_had_already_surpassed(self):
        with tempfile.TemporaryDirectory() as td:
            a=EpisodeExperience(observation_schema=MOTOR_V7_SCHEMA,action_sizes=(20,))
            a.add(obs(),(1,),obs(),player((('minecraft:oak_log',5),)),
                  player((('minecraft:oak_log',1),)),duration=24,reward=0.,done=False)
            a.add(obs(),(14,),obs(),player((('minecraft:oak_log',1),)),
                  player((('minecraft:oak_log',2),)),duration=72,reward=1.,done=False)
            a.save(td,'s',0)
            meta,data=read_episode(pathlib.Path(td)/'s'/'episode_000000000000.npz')
            matches=list(hindsight_goal_transitions(meta,data))
            self.assertEqual(len(matches),1)
            self.assertEqual(matches[0]['observation_index'],1)
            self.assertEqual(matches[0]['goal_item_id'],'minecraft:oak_log')

    def test_offline_trainer_reads_only_committed_self_play_and_resumes(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder)
            (root/'checkpoints').mkdir()
            (root/'logs').mkdir()
            torch.save(dict(
                stream='minecraft_seed_stream_v2_full64_interleaved_train_eval',
                metadata={'policy_schema':MOTOR_V7_SCHEMA},next_episode=2),
                root/'checkpoints'/'fresh_stream_source01.pt')
            a=EpisodeExperience(observation_schema=MOTOR_V7_SCHEMA,action_sizes=(20,))
            a.add(obs(),(14,),obs(),player(),
                  player((('minecraft:birch_log',1),)),duration=72,reward=3.,done=False)
            a.save(root/'data-local'/'experience','source01',0)
            # Future/uncommitted file MUST be excluded by PPO checkpoint cursor.
            a.save(root/'data-local'/'experience','source01',2)
            data=trainer.load_examples(root,'source01',
                         max_episodes=10,max_examples=20,lookback=4,seed=11)
            self.assertEqual(data[5]['episodes_scanned'],1)
            self.assertEqual(data[5]['training_examples'],1)
            self.assertEqual(data[5]['goals_by_item'],{'minecraft:birch_log':1})
            with mock.patch.object(trainer,'ROOT',root):
                trainer.main(['--source-run','source01','--run-tag','skills01',
                              '--epochs','1','--batch-size','1','--device','cpu'])
                saved=torch.load(root/'checkpoints'/'hindsight_skills01.pt',
                                 map_location='cpu',weights_only=True)
                self.assertEqual(saved['epochs'],1)
                with self.assertRaises(FileExistsError):
                    trainer.main(['--source-run','source01','--run-tag','skills01',
                                  '--epochs','2','--batch-size','1','--device','cpu'])
                trainer.main(['--source-run','source01','--run-tag','skills01',
                              '--epochs','2','--batch-size','1','--device','cpu','--resume'])
                saved=torch.load(root/'checkpoints'/'hindsight_skills01.pt',
                                 map_location='cpu',weights_only=True)
                self.assertEqual(saved['epochs'],2)
                self.assertEqual(saved['model']['updates'],2)

    def test_goal_frozen_evaluator_checks_source_motor_and_seed_split(self):
        import torch
        from fresh_seed_treadmill import parse_args,load_goal_control
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder)
            (root/'checkpoints').mkdir()
            torch.save(dict(metadata={
                'seed_root':917211,
                'reward_profile':'wood_sustain_v8',
                'policy_schema':MOTOR_V7_SCHEMA}),
                root/'checkpoints'/'fresh_stream_source01.pt')
            model=HindsightSkillModel(own_size=31,action_sizes=(20,),device='cpu')
            meta=dict(source_run='source01',
                source_observation_schema=MOTOR_V7_SCHEMA,
                own_size=31,action_sizes=(20,),seed=20261103)
            torch.save(dict(schema=SCHEMA,metadata=meta,
                            model=model.state()),root/'checkpoints'/'hindsight_skills01.pt')
            args=parse_args(['--run-tag','goaleval01','--mode','eval','--control','goal_sampled',
                    '--goal-model','skills01','--goal-item','minecraft:oak_log',
                    '--reward-profile','wood_sustain_v8','--decisions','24','--device','cpu'])
            ctrl=load_goal_control(root,args)
            act,logp,value=ctrl.act(obs())
            self.assertEqual(len(act),1)
            self.assertTrue(0<=act[0]<20)
            self.assertEqual((logp,value),(0.,0.))
            # A second frozen action RNG must change sampling, while the
            # learned checkpoint's weights remain exactly unchanged.
            original_rng=args.policy_seed
            seq_a=[ctrl.act(obs(),greedy=False)[0] for _ in range(25)]
            args.policy_seed=813777
            alternative=load_goal_control(root,args)
            seq_b=[alternative.act(obs(),greedy=False)[0] for _ in range(25)]
            self.assertNotEqual(seq_a,seq_b)
            args.policy_seed=original_rng
            replayed=load_goal_control(root,args)
            replayed.act(obs(),greedy=False)  # match initial controller's earlier test action
            self.assertEqual([replayed.act(obs(),greedy=False)[0] for _ in range(25)],seq_a)
            self.assertEqual(ctrl.model.act(obs(),'minecraft:oak_log',greedy=True),
                             alternative.model.act(obs(),'minecraft:oak_log',greedy=True))
            args.reward_profile='wood_motor_v7'
            with self.assertRaisesRegex(ValueError,'different source'):
                load_goal_control(root,args)
            with self.assertRaises(SystemExit):
                parse_args(['--run-tag','bad01','--mode','eval','--control','goal_sampled',
                            '--decisions','24'])

    def test_source_without_genuine_inventory_goals_fails(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder)
            (root/'checkpoints').mkdir()
            torch.save(dict(
                stream='minecraft_seed_stream_v2_full64_interleaved_train_eval',
                metadata={'policy_schema':MOTOR_V7_SCHEMA},next_episode=1),
                root/'checkpoints'/'fresh_stream_empty.pt')
            a=EpisodeExperience(observation_schema=MOTOR_V7_SCHEMA,action_sizes=(20,))
            a.add(obs(),(1,),obs(),player(),player(),duration=24,reward=4.,done=False)
            a.save(root/'data-local'/'experience','empty',0)
            with self.assertRaisesRegex(ValueError,'No TRAINING examples'):
                trainer.load_examples(root,'empty',max_episodes=1,
                                      max_examples=20,lookback=2,seed=0)

    def test_disk_retention_only_deletes_earliest_owned_archive_files(self):
        with tempfile.TemporaryDirectory() as td:
            directory=pathlib.Path(td)/'x'
            directory.mkdir()
            for index in range(3):
                (directory/f'episode_{index:012d}.npz').write_bytes(b'x'*650000)
            keep=directory/'do_not_touch.txt'
            keep.write_text('sentinel')
            deleted=compact_retention(td,'x',1_000_000)
            self.assertEqual(len(deleted),2)
            self.assertTrue((directory/'episode_000000000002.npz').exists())
            self.assertTrue(keep.exists())


if __name__=='__main__':unittest.main()
