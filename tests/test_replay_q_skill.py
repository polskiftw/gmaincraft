"""Pure safety/learning tests: player-only replay Q, no Minecraft fixtures."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments'))
from gmaincraft.experience_archive import EpisodeExperience,read_episode
from gmaincraft.natural_online_policy import (NaturalObservation,TEMPORAL_V9_SCHEMA,
    TEMPORAL_V9_SELF_FEATURES,MOTOR_V7_HEAD_SIZES)
from gmaincraft.player_knowledge import PlayerKnowledgeV1,OwnedItem
from gmaincraft.replay_q_skill import (ReplayQSkill,SCHEMA,CRACK_GOAL,
    goal_transitions,CRACK_RISE_INDEX,CONTINUITY_INDEX)
from gmaincraft.fresh_seed_schedule import SCHEMA as STREAM_SCHEMA
import train_replay_q as trainer
from fresh_seed_treadmill import parse_args,load_replay_q_control


def observation(*,crack=0.,continuity=0.):
    own=np.zeros(len(TEMPORAL_V9_SELF_FEATURES),np.float32)
    own[CRACK_RISE_INDEX]=crack
    own[CONTINUITY_INDEX]=continuity
    return NaturalObservation(own,np.zeros(48,np.uint8),np.zeros((24,48),np.uint8),
                              np.zeros((24,10),np.float32),np.zeros(24,np.float32))


def player(items=()):
    return PlayerKnowledgeV1('player_knowledge_v1',(0.,64.,0.),0.,0.,20.,
                             tuple(OwnedItem(i,k,v) for i,(k,v) in enumerate(items)),
                             (),(),())


def make_archive(root,name='source01',index=0):
    a=EpisodeExperience(observation_schema=TEMPORAL_V9_SCHEMA,action_sizes=MOTOR_V7_HEAD_SIZES)
    a.add(observation(),(12,),observation(crack=.2,continuity=1.),player(),player(),
          duration=6,reward=-100.,done=False)
    a.add(observation(),(14,),observation(crack=.2,continuity=0.),player(),
          player((('minecraft:oak_log',1),)),duration=72,reward=10000.,done=False)
    a.add(observation(),(4,),observation(),player((('minecraft:oak_log',1),)),
          player((('minecraft:oak_log',1),)),duration=24,reward=10000.,done=False)
    a.save(root/'data-local'/'experience',name,index)
    return read_episode(root/'data-local'/'experience'/name/f'episode_{index:012d}.npz')


def source_checkpoint(root,cursor=1):
    import torch
    (root/'checkpoints').mkdir(exist_ok=True)
    torch.save(dict(stream=STREAM_SCHEMA,metadata=dict(policy_schema=TEMPORAL_V9_SCHEMA,
               reward_profile='wood_temporal_v9',policy_action_heads=MOTOR_V7_HEAD_SIZES,
               seed_root=917211),next_episode=cursor),
               root/'checkpoints'/'fresh_stream_source01.pt')


class ReplayQTests(unittest.TestCase):
    def test_only_actual_player_crack_or_inventory_counts_as_achieved(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta,data=make_archive(Path(tmp))
            cracks=goal_transitions(meta,data,CRACK_GOAL)
            oak=goal_transitions(meta,data,'minecraft:oak_log')
            dirt=goal_transitions(meta,data,'minecraft:dirt')
            self.assertEqual([r['achieved'] for r in cracks],[True,False,False])
            self.assertEqual([r['achieved'] for r in oak],[False,True,False])
            self.assertEqual([r['achieved'] for r in dirt],[False]*3)
            self.assertTrue(cracks[0]['done'])
            self.assertTrue(oak[1]['done'])
            self.assertTrue(oak[-1]['done'])
            self.assertEqual(oak[0]['priority'],4.)
            self.assertEqual(cracks[1]['priority'],1.)
            self.assertNotIn('seed',oak[0]);self.assertNotIn('reward',oak[0])
            self.assertEqual(len(oak[0]['before_own']),len(TEMPORAL_V9_SELF_FEATURES))
            with self.assertRaises(ValueError):goal_transitions(meta,data,'hidden:world_oracle')
            bad_meta=dict(meta,observation_schema='other')
            with self.assertRaises(ValueError):goal_transitions(bad_meta,data,CRACK_GOAL)

    def test_checkpoint_gate_refuses_uncommitted_archive_and_empty_achievement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source_checkpoint(root,cursor=1)
            make_archive(root,index=0)
            make_archive(root,index=1)  # ahead of checkpoint -> excluded
            rows,stats=trainer.committed_rows(root,'source01','minecraft:oak_log',20)
            self.assertEqual(len(rows),3)
            self.assertEqual(stats['episodes_scanned'],1)
            self.assertEqual(stats['actual_successful_action_outcomes'],1)
            with self.assertRaisesRegex(ValueError,'No genuine achieved'):
                trainer.committed_rows(root,'source01','minecraft:dirt',20)

    def test_cuda_optional_cpu_learning_and_strict_frozen_sampling_seed(self):
        from gmaincraft.replay_q_skill import SCHEMA
        import torch
        m=ReplayQSkill(seed=31,device='cpu')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            meta,data=make_archive(root)
            rows=goal_transitions(meta,data,CRACK_GOAL)
            before=m.act(observation(),CRACK_GOAL,greedy=True)
            for _ in range(12):
                report=m.update(rows*3,CRACK_GOAL)
            self.assertTrue(np.isfinite(report['loss']))
            self.assertEqual(m.updates,12)
            state=m.state()
            self.assertEqual(state['schema'],SCHEMA)
            second=ReplayQSkill(seed=53,device='cpu')
            second.restore(state,optimizer=False)
            self.assertEqual(second.act(observation(),CRACK_GOAL,greedy=True),
                             m.act(observation(),CRACK_GOAL,greedy=True))
            with self.assertRaises(ValueError):m.act('raw world',CRACK_GOAL)
            with self.assertRaises(ValueError):m.update([],CRACK_GOAL)
            with self.assertRaises(ValueError):second.restore(dict(state,schema='other'))
            source_checkpoint(root)
            goal_metadata=dict(source_run='source01',source_observation_schema=TEMPORAL_V9_SCHEMA,
                               action_sizes=MOTOR_V7_HEAD_SIZES,goal=CRACK_GOAL,
                               source_seed_root=917211)
            torch.save(dict(schema=SCHEMA,model=state,metadata=goal_metadata),
                       root/'checkpoints'/'replay_q_skill01.pt')
            args=parse_args(['--mode','eval','--control','q_sampled','--run-tag','out01',
                '--q-model','skill01','--q-goal',CRACK_GOAL,
                '--reward-profile','wood_temporal_v9','--decisions','120','--device','cpu'])
            ctrl=load_replay_q_control(root,args)
            first=[ctrl.act(observation())[0] for _ in range(30)]
            args.policy_seed=81234
            alt=load_replay_q_control(root,args)
            alternate=[alt.act(observation())[0] for _ in range(30)]
            self.assertNotEqual(first,alternate)
            self.assertEqual(ctrl.act(observation(),greedy=True),
                             alt.act(observation(),greedy=True))
            args.q_goal='minecraft:oak_log'
            with self.assertRaisesRegex(ValueError,'goal/observation/motor'):
                load_replay_q_control(root,args)
            args.q_goal=CRACK_GOAL
            args.seed_root=2026
            with self.assertRaisesRegex(ValueError,'stream or player sensor'):
                load_replay_q_control(root,args)
            with self.assertRaises(SystemExit):
                parse_args(['--mode','eval','--control','q_sampled','--run-tag','bad1',
                            '--q-model','skill01','--reward-profile','wood_temporal_v9'])

    def test_real_data_cli_checkpoint_epoch_resume_without_overwrite(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            make_archive(root)
            source_checkpoint(root)
            with mock.patch.object(trainer,'ROOT',root):
                trainer.main(['--source-run','source01','--run-tag','q01','--goal',CRACK_GOAL,
                              '--device','cpu','--epochs','1','--updates-per-epoch','2',
                              '--batch-size','3'])
                output=root/'checkpoints'/'replay_q_q01.pt'
                one=torch.load(output,map_location='cpu',weights_only=True)
                self.assertEqual(one['epochs'],1)
                with self.assertRaises(FileExistsError):
                    trainer.main(['--source-run','source01','--run-tag','q01','--goal',CRACK_GOAL,
                                  '--device','cpu','--epochs','2'])
                trainer.main(['--source-run','source01','--run-tag','q01','--goal',CRACK_GOAL,
                              '--resume','--device','cpu','--epochs','2','--updates-per-epoch','2',
                              '--batch-size','3'])
                two=torch.load(output,map_location='cpu',weights_only=True)
                self.assertEqual(two['epochs'],2)
                self.assertEqual(two['model']['updates'],4)
                with self.assertRaisesRegex(ValueError,'metadata drift'):
                    trainer.main(['--source-run','source01','--run-tag','q01','--goal',CRACK_GOAL,
                                  '--resume','--device','cpu','--epochs','3',
                                  '--updates-per-epoch','4','--batch-size','3'])


# Exercises the real archive -> off-policy optimizer bridge, including the
# refusal to invent positive effects when a normal vanilla world made none.
class OnlineReplayTests(unittest.TestCase):
    def test_only_owned_positive_outcomes_enable_updates(self):
        from gmaincraft.online_q_replay import update_from_player_replay,recent_player_replay
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            make_archive(root,name='qtrain',index=2)
            model=ReplayQSkill(seed=71,device='cpu')
            positive=update_from_player_replay(model,root/'data-local'/'experience',
                'qtrain',2,goal=CRACK_GOAL,updates=4,batch_size=8,max_episodes=3)
            self.assertEqual(positive['updates'],4)
            self.assertEqual(positive['genuine_outcomes'],1)
            self.assertEqual(positive['last_optimizer_update'],4)
            self.assertEqual(model.updates,4)
            no_success=update_from_player_replay(model,root/'data-local'/'experience',
                'qtrain',2,goal='minecraft:dirt',updates=20,batch_size=8,max_episodes=3)
            self.assertEqual(no_success['updates'],0)
            self.assertEqual(model.updates,4)
            rows,episodes=recent_player_replay(root/'data-local'/'experience',
                                               'qtrain',2,goal=CRACK_GOAL)
            self.assertEqual(episodes,[2]);self.assertEqual(len(rows),3)
            self.assertEqual(update_from_player_replay(model,root,'qtrain',2,
                goal=CRACK_GOAL,updates=0,batch_size=8,max_episodes=3)['reason'],'disabled')
            with self.assertRaises(ValueError):
                update_from_player_replay(model,root,'qtrain',2,goal=CRACK_GOAL,
                                          updates=3,batch_size=2,max_episodes=3)

    def test_collector_run_metadata_and_initial_cursor_are_separate(self):
        from fresh_seed_treadmill import metadata,parse_args
        from gmaincraft.replay_q_skill import SCHEMA
        args=parse_args(['--mode','collect','--control','q_sampled','--run-tag','selftrain',
          '--q-model','skill01','--q-goal',CRACK_GOAL,'--reward-profile','wood_temporal_v9',
          '--collect-seed-start','4','--record-experience','--q-updates-per-world','4',
          '--decisions','24','--device','cpu'])
        self.assertEqual(metadata(args)['collector_start_index'],4)
        self.assertEqual(metadata(args)['collector_q_updates_per_world'],4)
        self.assertNotIn('collector_start_index',metadata(parse_args(['--run-tag','old_train'])))
        for invalid in (['--mode','collect','--control','q_sampled','--run-tag','selftrain',
          '--q-model','skill01','--q-goal',CRACK_GOAL,'--reward-profile','wood_temporal_v9',
          '--decisions','24'],
          ['--mode','eval','--control','q_sampled','--run-tag','eval01',
          '--q-model','skill01','--q-goal',CRACK_GOAL,'--reward-profile','wood_temporal_v9',
          '--decisions','24','--record-experience']):
            with self.assertRaises(SystemExit):parse_args(invalid)

class SnapshotTests(unittest.TestCase):
    def test_snapshot_of_only_committed_worlds_is_exclusive_and_loadable(self):
        import importlib.util
        import json
        import torch
        from gmaincraft.fresh_seed_schedule import SCHEMA as STREAM
        from fresh_seed_treadmill import parse_args,metadata,load_replay_q_control
        spec=importlib.util.spec_from_file_location('replay_snapshot',ROOT/'tools'/'export-replay-q-snapshot.py')
        mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'checkpoints').mkdir()
            (root/'logs').mkdir()
            collector_args=parse_args(['--mode','collect','--control','q_sampled',
                 '--run-tag','collector01','--q-model','firstq','--q-goal',CRACK_GOAL,
                 '--collect-seed-start','3','--reward-profile','wood_temporal_v9',
                 '--record-experience','--decisions','36','--device','cpu'])
            base_meta=metadata(collector_args)
            q=ReplayQSkill(seed=1,device='cpu')
            torch.save(dict(stream=STREAM,schema=SCHEMA,metadata=base_meta,
                            model=q.state(),next_episode=4,decisions=36,totals={}),
                            root/'checkpoints'/'fresh_stream_collector01.pt')
            (root/'logs'/'fresh_stream_collector01.jsonl').write_text(
               json.dumps(dict(stage='episode_committed',next_episode=4,
                               committed_decisions=36))+'\n'+
               json.dumps(dict(stage='uncommitted_pulse',episode=4))+'\n')
            report=mod.export_snapshot(root,'collector01','frozen01')
            self.assertEqual(report['committed_worlds'],1)
            self.assertEqual(report['q_updates'],0)
            args=parse_args(['--mode','eval','--control','q_sampled','--run-tag','eval01',
                  '--q-model','frozen01','--q-goal',CRACK_GOAL,
                  '--reward-profile','wood_temporal_v9','--device','cpu','--decisions','24'])
            self.assertEqual(len(load_replay_q_control(root,args).act(observation())[0]),1)
            with self.assertRaises(FileExistsError):mod.export_snapshot(root,'collector01','frozen01')
            with self.assertRaises(ValueError):mod.export_snapshot(root,'collector01','too_long_a_snapshot_export_name')
            (root/'logs'/'fresh_stream_collector01.jsonl').write_text(
                json.dumps(dict(stage='episode_committed',next_episode=5,
                                committed_decisions=72))+'\n')
            with self.assertRaisesRegex(ValueError,'disagree'):
                mod.export_snapshot(root,'collector01','newfreeze')

if __name__=='__main__':unittest.main()
