"""Trainable option chooser pure contract and real-file fail-closed tests.

Synthetic transitions here are tests of tensor and validation mechanics,
NEVER a source of claims or an artifact checkpoint in the actual Minecraft.
"""
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gmaincraft.natural_online_policy import NaturalObservation,TEMPORAL_V9_SELF_FEATURES
from gmaincraft.numeric_item_pickup import PickupReading
from gmaincraft.numeric_item_pickup_v2 import FEATURE_NAMES as ITEM_FEATURES
from gmaincraft.natural_option_q_v1 import (
    NaturalOptionQ,FEATURES,SCHEMA,option_features,own_observation_fingerprint,
    validate_real_row,reward_from_real_own_outcome)
spec=importlib.util.spec_from_file_location('natural_option_trainer',
    ROOT/'experiments/train_natural_option_offline.py')
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)


def reading():
    return PickupReading(np.zeros(len(ITEM_FEATURES),np.float32),None,False,0)


def miner():
    return NaturalObservation(np.zeros(len(TEMPORAL_V9_SELF_FEATURES),np.float32),
        np.zeros(48,np.uint8),np.zeros((24,48),np.uint8),
        np.zeros((24,10),np.float32),np.zeros(24,np.float32))


def row(*,option=0,gain=0):
    x=option_features(reading(),miner())
    return dict(schema=SCHEMA,before=x.tolist(),after=x.tolist(),
                option=option,ticks=6,wood_inventory_gain=gain,
                wood_mined=0,death=False,done=False)


class NaturalOptionQTests(unittest.TestCase):
    def test_only_own_camera_and_v9_features(self):
        x=option_features(reading(),miner())
        self.assertEqual(x.shape,(len(FEATURES),))
        self.assertEqual(x.dtype,np.float32)
        self.assertFalse(x.flags.writeable)
        self.assertEqual(len(x),len(ITEM_FEATURES)+8)
        self.assertFalse(any('seed' in name or 'world' in name or
            'correct_action' in name for name in FEATURES))
        fingerprint=own_observation_fingerprint(miner())
        self.assertEqual(len(fingerprint),24)
        self.assertEqual(fingerprint,own_observation_fingerprint(miner()))
        bad=miner();bad.own[0]=3.
        self.assertNotEqual(fingerprint,own_observation_fingerprint(bad))
        with self.assertRaises(TypeError):option_features(reading(),None)

    def test_real_outcome_contract_and_replay_checkpoint(self):
        self.assertEqual(reward_from_real_own_outcome(
            wood_inventory_gain=1,wood_mined=0,death=False,ticks=6),24.96)
        self.assertEqual(validate_real_row(row(option=1,gain=1))[1],1)
        with self.assertRaises(ValueError):validate_real_row(dict(row(),schema='imaginary_teacher'))
        with self.assertRaises(ValueError):validate_real_row(dict(row(),option=99))
        with self.assertRaises(ValueError):validate_real_row(dict(row(),wood_inventory_gain=-1))
        actor=NaturalOptionQ(seed=33)
        before=actor.net.layers[0].weight.detach().clone()
        examples=[row(option=i%2,gain=int(i%9==0)) for i in range(128)]
        out=actor.train_from_real_rows(examples,steps=4,batch_size=64)
        self.assertEqual(out['updates'],4)
        self.assertEqual(out['real_rows'],128)
        self.assertFalse(torch.equal(actor.net.layers[0].weight,before))
        restored=NaturalOptionQ(seed=44).restore(actor.state(),optimizer=False)
        self.assertEqual(restored.act(option_features(reading(),miner()),greedy=True),
                         actor.act(option_features(reading(),miner()),greedy=True))
        with self.assertRaises(ValueError):restored.restore(dict(schema='teacher_oracle'))

    def test_frozen_loader_requires_real_train_namespace_metadata(self):
        from fresh_seed_treadmill import parse_args,load_frozen_natural_option_control
        def args():
            return parse_args(['--run-tag','learnopt','--mode','eval','--control','q_sampled',
                '--q-model','qv2memf1104','--q-goal','effect:visible_wood_crack_progress_v2',
                '--pickup-model','hardrep1105b','--natural-option-model','fakeopt',
                '--pickup-selector','learned_option_v1',
                '--reward-profile','wood_temporal_v9','--device','cpu'])
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);(root/'checkpoints').mkdir()
            actor=NaturalOptionQ(seed=801)
            actor.train_from_real_rows([row(option=i%2,gain=int(i%9==0)) for i in range(128)],
                steps=3,batch_size=32)
            model=root/'checkpoints/natural_option_fakeopt.pt'
            metadata=dict(schema=SCHEMA,seed_namespace='train',optimizer_updates=3,
                          training_outcomes={'wood_gained':10},real_train_sources=['train_a'])
            torch.save(dict(schema=SCHEMA,metadata=metadata,learner=actor.state()),model)
            torch.manual_seed(817)
            rng_before=torch.get_rng_state().clone()
            restored=load_frozen_natural_option_control(root,args())
            self.assertTrue(torch.equal(rng_before,torch.get_rng_state()))
            self.assertEqual(restored.act(option_features(reading(),miner()),greedy=True),
                             actor.act(option_features(reading(),miner()),greedy=True))
            metadata['seed_namespace']='eval'
            torch.save(dict(schema=SCHEMA,metadata=metadata,learner=actor.state()),model)
            with self.assertRaises(ValueError):load_frozen_natural_option_control(root,args())

    def test_offline_training_refuses_no_real_files(self):
        with tempfile.TemporaryDirectory() as root,patch.object(trainer,'ROOT',pathlib.Path(root)):
            with self.assertRaises(FileNotFoundError):
                trainer.load_committed_real_trace('missing')
            with self.assertRaises(ValueError):
                trainer.load_committed_real_trace('../escape')
            log=pathlib.Path(root)/'logs';log.mkdir()
            (log/'fresh_eval_disallowed.jsonl').write_text(json.dumps({
                'stage':'config','mode':'eval','metadata':{'option_data_namespace':'eval'}})+'\n'+
                json.dumps({'stage':'completed','mode':'eval'})+'\n')
            with self.assertRaises(ValueError):
                trainer.load_committed_real_trace('disallowed')
            with self.assertRaises(ValueError):
                trainer.train(tag='trained',sources=['disallowed'])

if __name__=='__main__':unittest.main()
