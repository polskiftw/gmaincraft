"""V2 only: no script/teacher; player knowledge, rare positive memory, resume."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments'))
from gmaincraft.experience_archive import EpisodeExperience,read_episode
from gmaincraft.fresh_seed_schedule import SCHEMA as STREAM_SCHEMA
from gmaincraft.natural_online_policy import (NaturalObservation,
    TEMPORAL_V9_SCHEMA,TEMPORAL_V9_SELF_FEATURES,MOTOR_V7_HEAD_SIZES)
from gmaincraft.player_knowledge import PlayerKnowledgeV1,OwnedItem
from gmaincraft.replay_q_skill import ReplayQSkill,CRACK_GOAL,SCHEMA as Q_V1_SCHEMA
from gmaincraft.replay_q_rare_v2 import (RareOutcomeQSkill,WOOD_CRACK_GOAL,
    SCHEMA,WOOD_INDEX,wood_crack_transitions)
from gmaincraft.online_q_rare_v2 import (buckets,committed_prior_success_source,
    new_world_buckets,learn_between_real_worlds)
from fresh_seed_treadmill import parse_args,metadata,load_replay_q_control
import train_replay_q_rare_v2 as trainer


def obs(wood=0.,rise=0.,same=0.):
    own=np.zeros(len(TEMPORAL_V9_SELF_FEATURES),dtype=np.float32)
    own[WOOD_INDEX]=wood
    own[TEMPORAL_V9_SELF_FEATURES.index('crack_stage_rose_same_target_div_10')]=rise
    own[TEMPORAL_V9_SELF_FEATURES.index('same_vanilla_crosshair_target_as_previous')]=same
    return NaturalObservation(own,np.zeros(48,np.uint8),
                              np.zeros((24,48),np.uint8),
                              np.zeros((24,10),np.float32),np.zeros(24,np.float32))


def player(items=()):
    return PlayerKnowledgeV1('player_knowledge_v1',(0.,64.,0.),0.,0.,20.,
                             tuple(OwnedItem(i,k,v) for i,(k,v) in enumerate(items)),
                             (),(),())


def archive(root,run,index):
    a=EpisodeExperience(observation_schema=TEMPORAL_V9_SCHEMA,action_sizes=MOTOR_V7_HEAD_SIZES)
    # These artificial OUTCOMES occur exclusively in PURE TESTS, never
    # inserted into training data or actual game checkpoints.
    a.add(obs(wood=1.),(14,),obs(wood=1.,rise=.3,same=1.),player(),player(),
          duration=72,reward=5000.,done=False)
    a.add(obs(wood=1.),(11,),obs(wood=1.),player(),player(),duration=6,
          reward=9000.,done=False)
    a.add(obs(wood=0.),(13,),obs(wood=0.,rise=.5,same=1.),player(),player(),
          duration=24,reward=7777.,done=False)
    a.add(obs(wood=0.),(2,),obs(wood=0.),player(),player(),duration=24,
          reward=1111.,done=False)
    return a.save(root/'data-local'/'experience',run,index)


def source(root,tag='prior01',cursor=1):
    import torch
    (root/'checkpoints').mkdir(exist_ok=True)
    return torch.save(dict(stream=STREAM_SCHEMA,schema='irrelevant',next_episode=cursor,
        metadata=dict(kind='gmaincraft_q_collect_v1',policy_schema=TEMPORAL_V9_SCHEMA,
        policy_action_heads=MOTOR_V7_HEAD_SIZES,reward_profile='wood_temporal_v9',
        seed_root=917211)),root/'checkpoints'/f'fresh_stream_{tag}.pt')


class RareV2Tests(unittest.TestCase):
    def test_genuine_wood_crack_only_not_reward_or_nonwood_cracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            archive(root,'prior01',0)
            meta,data=read_episode(root/'data-local'/'experience'/'prior01'/'episode_000000000000.npz')
            rows=wood_crack_transitions(meta,data)
            self.assertEqual([r['achieved'] for r in rows],[True,False,False,False])
            self.assertEqual([r['done'] for r in rows],[True,False,False,True])
            self.assertEqual(rows[0]['action'],14)
            self.assertNotIn('reward',rows[0])
            self.assertNotIn('seed',rows[0])
            self.assertEqual(len(buckets(rows)[0]),1)
            self.assertEqual(len(buckets(rows)[1]),1)
            self.assertEqual(len(buckets(rows)[2]),2)
            # A change only in nonwood crack stage is NOT a positive outcome.
            self.assertFalse(rows[2]['achieved'])
            self.assertEqual(rows[0]['priority'],20.)
            self.assertEqual(rows[2]['priority'],1.)

    def test_versioned_actor_initialized_only_from_own_prior_checkpoint(self):
        original=ReplayQSkill(seed=11,device='cpu')
        saved=dict(schema='player_known_conservative_goal_q_v1',
                   metadata=dict(goal=CRACK_GOAL),model=original.state())
        m=RareOutcomeQSkill(seed=15,device='cpu')
        m.initialize_from_selfplay(saved)
        st=m.state()
        self.assertEqual(st['schema'],SCHEMA)
        self.assertEqual(st['goal'],WOOD_CRACK_GOAL)
        self.assertEqual(m.act(obs(wood=1.),WOOD_CRACK_GOAL,greedy=True),
                         RareOutcomeQSkill(seed=13,device='cpu').initialize_from_selfplay(saved).act(
                             obs(wood=1.),WOOD_CRACK_GOAL,greedy=True))
        with self.assertRaises(ValueError):m.act(obs(wood=1.),CRACK_GOAL)
        anchor=[x.detach().clone() for x in m.anchor.parameters()]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            archive(root,'prior01',0)
            meta,data=read_episode(root/'data-local'/'experience'/'prior01'/'episode_000000000000.npz')
            batch=wood_crack_transitions(meta,data)*4
            for i in range(5):
                metrics=m.update(batch,WOOD_CRACK_GOAL)
            self.assertEqual(metrics['updates'],5)
            self.assertEqual(metrics['positive_own_actions'],4)
            self.assertTrue(np.isfinite(metrics['anchor_kl']))
        self.assertTrue(all(m.torch.equal(a,b) for a,b in zip(anchor,m.anchor.parameters())))
        restored=RareOutcomeQSkill(seed=99,device='cpu')
        restored.restore(m.state(),optimizer=False)
        self.assertEqual(restored.act(obs(wood=1.),WOOD_CRACK_GOAL,greedy=True),
                         m.act(obs(wood=1.),WOOD_CRACK_GOAL,greedy=True))
        with self.assertRaises(ValueError):restored.restore(dict(m.state(),schema='wrong'))
        with self.assertRaises(ValueError):m.initialize_from_selfplay(dict(saved,metadata=dict(goal='minecraft:oak_log')))

    def test_protected_success_replay_training_when_new_episode_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source(root,cursor=1)
            archive(root,'prior01',0)
            a=EpisodeExperience(observation_schema=TEMPORAL_V9_SCHEMA,action_sizes=MOTOR_V7_HEAD_SIZES)
            a.add(obs(wood=1.),(0,),obs(wood=1.),player(),player(),duration=6,reward=0.,done=False)
            a.add(obs(wood=0.),(3,),obs(wood=0.),player(),player(),duration=6,reward=0.,done=False)
            a.save(root/'data-local'/'experience','newrun',1)
            prior,stats=committed_prior_success_source(str(root.resolve()),'prior01')
            self.assertEqual(len(prior[0]),1)
            self.assertEqual(stats['source_cursor'],1)
            self.assertEqual(len(new_world_buckets(root/'data-local'/'experience','newrun',1,
                                                   recent_worlds=3)[0][0]),0)
            model=RareOutcomeQSkill(seed=5,device='cpu')
            result=learn_between_real_worlds(model,root,'newrun',1,source_run='prior01',
                                              updates=3,batch_size=12,recent_worlds=3)
            self.assertEqual(result['source_verified_successes'],1)
            self.assertEqual(result['new_own_successes'],0)
            self.assertEqual(result['updates'],3)
            self.assertEqual(model.updates,3)
            with self.assertRaisesRegex(ValueError,'same immutable prior source'):
                learn_between_real_worlds(model,root,'prior01',1,source_run='prior01',
                                          updates=1,batch_size=12,recent_worlds=3)
            with self.assertRaises(ValueError):
                learn_between_real_worlds(model,root,'newrun',1,source_run='prior01',
                                          updates=1,batch_size=11,recent_worlds=3)

    def test_collector_unchanged_v1_metadata_and_source_seed_nonreuse(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source(root,tag='old01',cursor=7)
            m=RareOutcomeQSkill(device='cpu')
            anchor=ReplayQSkill(device='cpu')
            m.initialize_from_selfplay(dict(schema=Q_V1_SCHEMA,metadata=dict(goal=CRACK_GOAL),
                                            model=anchor.state()))
            metadata_q=dict(source_run='old01',source_observation_schema=TEMPORAL_V9_SCHEMA,
                            action_sizes=MOTOR_V7_HEAD_SIZES,goal=WOOD_CRACK_GOAL,
                            source_seed_root=917211)
            torch.save(dict(schema=SCHEMA,model=m.state(),metadata=metadata_q),
                       root/'checkpoints'/'replay_q_v2model.pt')
            base=['--mode','collect','--control','q_sampled','--run-tag','v2collector',
                  '--q-model','v2model','--q-goal',WOOD_CRACK_GOAL,
                  '--reward-profile','wood_temporal_v9','--record-experience',
                  '--decisions','24','--device','cpu']
            too_early=parse_args(base+['--collect-seed-start','6'])
            with self.assertRaisesRegex(ValueError,'never-seen TRAIN indices'):
                load_replay_q_control(root,too_early)
            valid=parse_args(base+['--collect-seed-start','7'])
            ctrl=load_replay_q_control(root,valid)
            self.assertEqual(ctrl.prior_source_run,'old01')
            self.assertEqual(ctrl.schema,SCHEMA)
            self.assertEqual(metadata(valid)['kind'],'gmaincraft_q_collect_v2_rare_outcome')
            self.assertEqual(metadata(valid)['collector_q_v2_anchor_weight'],2.)
            old=parse_args(['--run-tag','legacy','--mode','collect','--control','q_sampled',
              '--q-model','old01','--q-goal',CRACK_GOAL,'--reward-profile','wood_temporal_v9',
              '--collect-seed-start','2','--record-experience','--decisions','24'])
            self.assertFalse(any('_q_v2_' in key for key in metadata(old)))
            with self.assertRaises(SystemExit):parse_args(base+['--q-v2-anchor-weight','15'])

    def test_offline_training_snapshot_and_resume(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'logs').mkdir()
            source(root,cursor=1)
            archive(root,'prior01',0)
            old=ReplayQSkill(device='cpu')
            torch.save(dict(schema=Q_V1_SCHEMA,model=old.state(),metadata=dict(
              goal=CRACK_GOAL,source_seed_root=917211)),
              root/'checkpoints'/'replay_q_anchor01.pt')
            with mock.patch.object(trainer,'ROOT',root):
                cmd=['--source-run','prior01','--anchor-model','anchor01',
                     '--run-tag','new01','--device','cpu','--batch-size','12',
                     '--updates-per-epoch','2','--epochs','1','--anchor-weight','2']
                trainer.main(cmd)
                saved=torch.load(root/'checkpoints'/'replay_q_new01.pt',map_location='cpu',weights_only=True)
                self.assertEqual(saved['schema'],SCHEMA)
                self.assertEqual(saved['epochs'],1)
                with self.assertRaises(FileExistsError):trainer.main(cmd)
                cmd[cmd.index('1')]='2'  # change cumulative epoch target
                trainer.main(cmd+['--resume'])
                last=torch.load(root/'checkpoints'/'replay_q_new01.pt',map_location='cpu',weights_only=True)
                self.assertEqual(last['epochs'],2)
                self.assertEqual(last['model']['updates'],4)

if __name__=='__main__':unittest.main()

class ChainedExperienceTests(unittest.TestCase):
    def test_two_separate_completed_train_sources_retained_forever(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source(root,tag='prior01',cursor=1)
            source(root,tag='second01',cursor=2)
            import torch
            cp=root/'checkpoints'/'fresh_stream_second01.pt'
            record=torch.load(cp,map_location='cpu',weights_only=True)
            record['metadata']['collector_start_index']=1
            torch.save(record,cp)
            archive(root,'prior01',0)
            archive(root,'second01',1)
            a=EpisodeExperience(observation_schema=TEMPORAL_V9_SCHEMA,action_sizes=MOTOR_V7_HEAD_SIZES)
            a.add(obs(wood=0.),(4,),obs(wood=0.),player(),player(),duration=24,
                  reward=9.,done=False)
            a.save(root/'data-local'/'experience','third01',2)
            m=RareOutcomeQSkill(seed=10,device='cpu')
            result=learn_between_real_worlds(m,root,'third01',2,source_run='prior01',
                    additional_source_run='second01',updates=3,batch_size=12)
            self.assertEqual(result['source_verified_successes'],2)
            self.assertEqual(result['new_own_successes'],0)
            self.assertEqual(result['additional_source_run'],'second01')
            self.assertEqual(m.updates,3)
            with self.assertRaisesRegex(ValueError,'duplicate'):
                learn_between_real_worlds(m,root,'third01',2,source_run='prior01',
                    additional_source_run='prior01',updates=2,batch_size=12)

    def test_v2_collector_export_carries_parent_replay_provenance(self):
        import importlib.util
        import torch
        spec=importlib.util.spec_from_file_location('replay_snapshot',
                                      ROOT/'tools'/'export-replay-q-snapshot.py')
        mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'logs').mkdir()
            source(root,tag='prior01',cursor=1)
            source(root,tag='collector01',cursor=3)
            initial=RareOutcomeQSkill(seed=31,device='cpu')
            torch.save(dict(schema=SCHEMA,metadata=dict(goal=WOOD_CRACK_GOAL,
                source_run='prior01',source_seed_root=917211),model=initial.state()),
                root/'checkpoints'/'replay_q_startv2.pt')
            ckpt=torch.load(root/'checkpoints'/'fresh_stream_collector01.pt',
                            map_location='cpu',weights_only=True)
            ckpt.update(schema=SCHEMA,model=initial.state(),decisions=36,
                        metadata=dict(ckpt['metadata'],kind='gmaincraft_q_collect_v2_rare_outcome',
                            collector_q_model='startv2',collector_q_goal=WOOD_CRACK_GOAL,
                            collector_start_index=2))
            torch.save(ckpt,root/'checkpoints'/'fresh_stream_collector01.pt')
            (root/'logs'/'fresh_stream_collector01.jsonl').write_text(json.dumps(dict(
                stage='episode_committed',next_episode=3,committed_decisions=36))+'\n')
            result=mod.export_snapshot(root,'collector01','chainedv2')
            saved=torch.load(root/'checkpoints'/'replay_q_chainedv2.pt',
                             map_location='cpu',weights_only=True)
            self.assertEqual(saved['schema'],SCHEMA)
            self.assertEqual(saved['metadata']['source_run'],'collector01')
            self.assertEqual(saved['metadata']['protected_prior_source_run'],'prior01')
            args=parse_args(['--mode','collect','--control','q_sampled',
                  '--run-tag','newthird','--q-model','chainedv2','--q-goal',WOOD_CRACK_GOAL,
                  '--reward-profile','wood_temporal_v9','--record-experience',
                  '--collect-seed-start','3','--decisions','24','--device','cpu'])
            ctrl=load_replay_q_control(root,args)
            self.assertEqual(ctrl.prior_source_run,'prior01')
            self.assertEqual(ctrl.additional_prior_source_run,'collector01')
            with self.assertRaises(FileExistsError):mod.export_snapshot(root,'collector01','chainedv2')
