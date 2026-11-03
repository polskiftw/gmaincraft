"""V5 exact player/crosshair+RGB schema, no privileged world leakage."""
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.agent_interface import MCioAgentBridge,MultimodalStatusView
from gmaincraft.player_knowledge import PlayerKnowledgeV1
from gmaincraft.mcio_status import PlayerStatusV1,PlayerStatusOption,SCHEMA as STATUS_SCHEMA
from gmaincraft.status_factorized import (StatusFactorizedPPOTrainer,
    StatusFactorTransition,encode_status_knowledge,FEATURE_NAMES,FEATURE_SIZE,
    BLOCK_TOKEN_LENGTH,BLOCK_TOKENIZER,SCHEMA)
from mcio_ctrl.network import ObservationPacket


def fake_player_packet():
    return SimpleNamespace(player_pos=(1.5,65.,-0.5),player_yaw=0.,player_pitch=30.,health=18.,
        inventory_main=[SimpleNamespace(slot=0,id='minecraft:oak_log',count=2)],
        inventory_armor=[],inventory_offhand=[],world_seed=181818,hidden_ore=[1,2],
        get_frame=lambda:np.zeros((180,320,3),dtype=np.uint8))


def fake_status(*,block_id='minecraft:oak_log'):
    hit=block_id is not None
    option=PlayerStatusOption(schema=STATUS_SCHEMA,hunger=19,saturation=4.5,
        armor_points=0,air_supply=300,selected_hotbar_slot=0,
        dimension='minecraft:overworld',targeted_block_id=block_id,
        targeted_block_pos=[1,66,2] if hit else None,
        targeted_block_face='north' if hit else None,
        targeted_hit_distance=2.5 if hit else -1.)
    return PlayerStatusV1.from_packet(ObservationPacket(options=[option]))


class V5Tests(unittest.TestCase):
    def setUp(self):
        self.player=PlayerKnowledgeV1.from_packet(fake_player_packet())

    def test_stable_player_status_and_exact_full_block_identifier(self):
        status=fake_status()
        num,tokens=encode_status_knowledge(self.player,status)
        self.assertEqual(num.shape,(FEATURE_SIZE,))
        self.assertEqual(tokens.shape,(BLOCK_TOKEN_LENGTH,))
        self.assertEqual(len(FEATURE_NAMES),FEATURE_SIZE)
        self.assertTrue(np.isfinite(num).all())
        self.assertFalse(num.flags.writeable)
        self.assertFalse(tokens.flags.writeable)
        encoded=bytes(int(v-1) for v in tokens if v).decode('utf-8')
        self.assertEqual(encoded,'minecraft:oak_log')
        self.assertEqual(int(num[FEATURE_NAMES.index('crosshair_has_block')]),1)
        self.assertAlmostEqual(float(num[FEATURE_NAMES.index('hunger_per_20')]),.95,places=6)
        self.assertAlmostEqual(float(num[FEATURE_NAMES.index('crosshair_rel_z_per_8')]),2.5/8,places=6)
        self.assertEqual(float(num[FEATURE_NAMES.index('crosshair_face_north')]),1.)
        self.assertFalse('hidden_ore' in repr(status))

    def test_miss_and_different_identifier(self):
        miss,tokens=encode_status_knowledge(self.player,fake_status(block_id=None))
        self.assertEqual(int(miss[FEATURE_NAMES.index('crosshair_has_block')]),0)
        self.assertTrue(np.all(tokens==0))
        for key in FEATURE_NAMES:
            if key.startswith(('crosshair_rel','crosshair_face','crosshair_distance')):
                self.assertEqual(float(miss[FEATURE_NAMES.index(key)]),0.)
        _,other=encode_status_knowledge(self.player,fake_status(block_id='minecraft:stone'))
        self.assertFalse(np.array_equal(other,encode_status_knowledge(self.player,fake_status())[1]))
        self.assertEqual(bytes(int(v-1) for v in other if v).decode('utf-8'),'minecraft:stone')

    def test_bridge_v5_opt_in_only(self):
        packet=fake_player_packet()
        actual=ObservationPacket(options=[PlayerStatusOption(schema=STATUS_SCHEMA,
            hunger=19,saturation=4.5,armor_points=0,air_supply=300,
            selected_hotbar_slot=0,dimension='minecraft:overworld',
            targeted_block_id='minecraft:oak_log',targeted_block_pos=[1,66,2],
            targeted_block_face='north',targeted_hit_distance=2.5)])
        # The bridge deliberately keeps all prior actions/tests unchanged.
        packet.get_option=actual.get_option
        bridge=object.__new__(MCioAgentBridge)
        bridge.packet=packet
        bridge._stats={}
        view=bridge.multimodal_status_view()
        self.assertIsInstance(view,MultimodalStatusView)
        self.assertEqual(tuple(vars(view)),('pixels','player','status'))
        self.assertFalse(hasattr(view,'world_seed'))
        self.assertFalse(hasattr(view,'hidden_ore'))
        self.assertFalse(view.pixels.flags.writeable)
        self.assertEqual(view.status.crosshair_block.block_id,'minecraft:oak_log')

    def test_network_reacts_to_full_status_and_training_stays_schema_isolated(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Optional PyTorch not installed in CI')
        torch.set_num_threads(2)
        agent=StatusFactorizedPPOTrainer(device='cpu',seed=470,update_epochs=1)
        image=np.zeros((6,84,84),dtype=np.uint8)
        features,ids=encode_status_knowledge(self.player,fake_status())
        _,other_ids=encode_status_knowledge(self.player,fake_status(block_id='minecraft:stone'))
        with torch.no_grad():
            first=agent.net(agent._images(image[None]),agent._features(features[None]),
                            agent._tokens(ids[None]))
            second=agent.net(agent._images(image[None]),agent._features(features[None]),
                             agent._tokens(other_ids[None]))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(first[0],second[0])))
        factors,logp,value=agent.act(image,features,ids,greedy=False)
        transition=StatusFactorTransition(image,features.copy(),ids.copy(),factors,
                                          1.,value,logp,6,True)
        old={k:v.clone() for k,v in agent.net.state_dict().items()}
        output=agent.update([transition])
        self.assertEqual(output['transitions'],1)
        self.assertTrue(any(not torch.equal(v,agent.net.state_dict()[k]) for k,v in old.items()))
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)/'status_model.pt'
            agent.save(path,dict(schema=SCHEMA,feature_names=FEATURE_NAMES,
                                 status_schema=STATUS_SCHEMA,block_tokenizer=BLOCK_TOKENIZER,
                                 block_token_length=BLOCK_TOKEN_LENGTH))
            other=StatusFactorizedPPOTrainer(device='cpu',seed=999)
            metadata=other.load_weights(path)
            self.assertEqual(metadata['schema'],SCHEMA)
            self.assertEqual(agent.act(image,features,ids,greedy=True)[0],
                             other.act(image,features,ids,greedy=True)[0])

if __name__=='__main__':unittest.main()
