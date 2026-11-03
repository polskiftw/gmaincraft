"""V4 contract: no world leakage, stable encoding, and two-channel neural use."""
import pathlib
import sys
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.agent_interface import MCioAgentBridge,MultimodalView
from gmaincraft.player_knowledge import PlayerKnowledgeV1
from gmaincraft.memory_factorized import (encode_player_knowledge,FEATURE_NAMES,
    FEATURE_SIZE,MemoryFactorizedPPOTrainer,MemoryFactorTransition,SCHEMA)


def fake_packet():
    return SimpleNamespace(player_pos=(16.25,64.0,-3.5),player_yaw=90.,
        player_pitch=25.,health=14.,
        inventory_main=[SimpleNamespace(slot=0,id='minecraft:oak_log',count=3)],
        inventory_armor=[],inventory_offhand=[],world_seed=12234234,
        undiscovered_stronghold=(33,12,21),hidden_block_ids=['diamond_ore'],
        get_frame=lambda:np.zeros((180,320,3),dtype=np.uint8))


class MemoryPolicyTests(unittest.TestCase):
    def test_numeric_features_and_hidden_world_exclusion(self):
        stats={'minecraft:mined':{'minecraft:oak_log':2},
               'minecraft:picked_up':{'minecraft:oak_log':1},
               'minecraft:custom':{'minecraft:hidden':93}}
        p=PlayerKnowledgeV1.from_packet(fake_packet(),stats=stats)
        f=encode_player_knowledge(p)
        self.assertEqual(f.shape,(FEATURE_SIZE,))
        self.assertEqual(len(FEATURE_NAMES),FEATURE_SIZE)
        self.assertFalse(f.flags.writeable)
        self.assertAlmostEqual(float(f[FEATURE_NAMES.index('yaw_sin')]),1.,places=5)
        self.assertAlmostEqual(float(f[FEATURE_NAMES.index('owned_logs_per_32')]),3/32,places=5)
        self.assertAlmostEqual(float(f[FEATURE_NAMES.index('mined_logs_per_32')]),2/32,places=5)
        self.assertFalse(hasattr(p,'world_seed'))
        self.assertFalse(hasattr(p,'undiscovered_stronghold'))
        self.assertFalse(hasattr(p,'hidden_block_ids'))
        self.assertFalse('minecraft:hidden' in repr(p))

    def test_unknown_stat_distinct_from_zero_and_roundtrip(self):
        p=PlayerKnowledgeV1.from_packet(fake_packet())
        missing=encode_player_knowledge(p)
        known=encode_player_knowledge(PlayerKnowledgeV1.from_packet(
            fake_packet(),stats={'minecraft:mined':{'minecraft:oak_log':0}}))
        self.assertEqual(float(missing[FEATURE_NAMES.index('mined_stats_available')]),0.)
        self.assertEqual(float(known[FEATURE_NAMES.index('mined_stats_available')]),1.)
        self.assertEqual(float(known[FEATURE_NAMES.index('mined_logs_per_32')]),0.)
        with self.assertRaises(ValueError):encode_player_knowledge(replace(p,schema='other'))

    def test_canonical_bridge_multimodal_view_contains_only_own_state_and_rgb(self):
        bridge=object.__new__(MCioAgentBridge)
        bridge.packet=fake_packet()
        bridge._stats={'minecraft:mined':{'minecraft:oak_log':1}}
        view=bridge.multimodal_view()
        self.assertIsInstance(view,MultimodalView)
        self.assertEqual(tuple(vars(view)),('pixels','player'))
        self.assertFalse(view.pixels.flags.writeable)
        self.assertEqual(view.player.player_xyz,(16.25,64.0,-3.5))
        self.assertFalse(hasattr(view,'world_seed'))
        self.assertFalse(hasattr(view,'packet'))
        self.assertFalse(hasattr(view.player,'undiscovered_stronghold'))

    def test_random_weight_network_consumes_numeric_channel_and_ppo_updates(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Optional GPU torch not installed in base CI')
        torch.set_num_threads(2)
        model=MemoryFactorizedPPOTrainer(device='cpu',seed=203,update_epochs=1)
        image=np.zeros((6,84,84),dtype=np.uint8)
        feature=np.zeros(FEATURE_SIZE,dtype=np.float32)
        altered=feature.copy()
        altered[FEATURE_NAMES.index('yaw_sin')]=1.
        with torch.no_grad():
            original=model.net(model._images(image[None]),model._features(feature[None]))
            changed=model.net(model._images(image[None]),model._features(altered[None]))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(original[0],changed[0])))
        factors,logp,value=model.act(image,feature,greedy=False)
        transition=MemoryFactorTransition(image,feature.copy(),factors,1.,value,logp,6,True)
        before={k:v.clone() for k,v in model.net.state_dict().items()}
        result=model.update([transition])
        self.assertEqual(result['transitions'],1)
        self.assertTrue(any(not torch.equal(v,model.net.state_dict()[k]) for k,v in before.items()))
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)/'memory.pt'
            model.save(path,dict(schema=SCHEMA,feature_names=FEATURE_NAMES))
            restored=MemoryFactorizedPPOTrainer(device='cpu',seed=1)
            self.assertEqual(restored.load_weights(path)['schema'],SCHEMA)
            self.assertEqual(model.act(image,feature,greedy=True)[0],
                             restored.act(image,feature,greedy=True)[0])

if __name__=='__main__': unittest.main()
