"""Independent staged item curriculum must keep a strict player-only contract."""
import json
import pathlib
import sys
import tempfile
import unittest
from dataclasses import replace

import torch

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments'))
from train_numeric_item_pickup import (
    parse_args,metadata,own_oak,publish_checkpoint,restore_checkpoint,
    beyond_practice_boundary,reset_case_safely)
from gmaincraft.numeric_item_pickup import NumericPickupActor,SCHEMA
from gmaincraft.player_knowledge import (
    OwnedItem,PlayerKnowledgeV1,SCHEMA as PLAYER_SCHEMA)


class ItemRunnerTest(unittest.TestCase):
    def test_versioned_stage_and_frozen_eval_controls(self):
        a=parse_args(['--run-tag','itemtrain','--episodes','50'])
        b=parse_args(['--run-tag','itemeval','--mode','eval',
                      '--model-tag','itemtrain','--episodes','10',
                      '--control','greedy'])
        self.assertEqual(metadata(a)['schema'],SCHEMA)
        self.assertEqual(metadata(a)['action_names'],metadata(b)['action_names'])
        self.assertTrue(metadata(a)['no_admin_commands_during_scored_play'])
        self.assertTrue(metadata(b)['memory_only_no_rgb'])
        self.assertNotIn('screenshot',metadata(a)['observation'])
        self.assertNotEqual(a.case_seed,0)

    def test_actual_own_inventory_not_mining_stat(self):
        p=PlayerKnowledgeV1(PLAYER_SCHEMA,(0.,70.,0.),0.,30.,20.,
                            (OwnedItem(0,'minecraft:oak_log',3),
                             OwnedItem(1,'minecraft:oak_planks',12)),
                            (),(),(('minecraft:mined','minecraft:oak_log',1000),))
        self.assertEqual(own_oak(p),3)

    def test_player_observed_start_boundary_not_hidden_fixture_target(self):
        own_start=(12.,70.,23.)
        self.assertFalse(beyond_practice_boundary(own_start,(17.,70.,23.)))
        self.assertTrue(beyond_practice_boundary(own_start,(19.,70.,23.)))
        self.assertTrue(beyond_practice_boundary(own_start,(12.,66.,23.)))
        self.assertFalse(beyond_practice_boundary(own_start,(12.,70.,23.)))

    def test_fixture_retry_is_bounded_and_only_before_scored_episode(self):
        from types import SimpleNamespace
        class Arena:
            def __init__(self):
                self.calls=0
                self.client=SimpleNamespace(scoring=True,advance=lambda n:None)
            def reset(self,case):
                self.calls+=1
                if self.calls==1:
                    raise RuntimeError('Not teleported to fresh scenario')
                return 'real-observation'
        a=Arena()
        self.assertEqual(reset_case_safely(a,object()),('real-observation',1))
        self.assertFalse(a.client.scoring)
        self.assertEqual(a.calls,2)

    def test_uncommitted_rows_recover_only_from_atomic_model(self):
        args=parse_args(['--run-tag','a','--episodes','12'])
        brain=NumericPickupActor(seed=4,device='cpu')
        with tempfile.TemporaryDirectory() as d:
            folder=pathlib.Path(d)
            model=folder/'pickup.pt';journal=folder/'journal.jsonl'
            journal.write_text(json.dumps(dict(stage='config',metadata=metadata(args)))+'\n')
            latest=dict(stage='episode_committed',episode=0,
                        actual_acquired=False,own_inventory_gain=0)
            publish_checkpoint(model,dict(schema=SCHEMA,metadata=metadata(args),
                    brain=brain.state(),next_episode=1,latest_result=latest))
            saved,repair=restore_checkpoint(model,journal,args)
            self.assertEqual(saved['next_episode'],1)
            self.assertEqual(repair['episode'],0)
            with journal.open('a') as stream:
                stream.write(json.dumps(latest)+'\n')
            _,repair=restore_checkpoint(model,journal,args)
            self.assertIsNone(repair)
            with self.assertRaises(FileExistsError):
                publish_checkpoint(model,dict(schema='x')) if model.with_suffix('.pt.pending').exists() else (
                    model.with_suffix('.pt.pending').touch(),
                    publish_checkpoint(model,dict(schema='x')))


if __name__=='__main__':unittest.main()
