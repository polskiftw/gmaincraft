"""Test that Gamescope crashes are nonfatal only with true committed evidence."""
from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util,json,sys,unittest,fcntl
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pickup_verifier',ROOT/'tools/verify-numeric-pickup-v2.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class VerifyPickupTest(unittest.TestCase):
    def test_completed_training_checkpoint_and_missing_commit_are_detected(self):
        with TemporaryDirectory() as d,patch.object(module,'ROOT',Path(d)):
            journal,checkpoint,lock=module.paths('goodrun')
            journal.parent.mkdir();checkpoint.parent.mkdir()
            meta={'schema':'numeric_pickup_egocentric_replay_q_v2','action_names':['forward6','forward24']}
            config={'stage':'config','mode':'train','metadata':meta,'frozen_source':None}
            committed={'stage':'episode_committed','episode':0,'actual_acquired':True,
                'own_inventory_gain':1,'final_held_oak':1,'scored_controller_commands':0,
                'actor_updates':12,'item_visible_at_first_action':True}
            summary={'stage':'completed','episodes':1,'successes':1,
                'inventory_items':1,'optimizer_updates':12}
            with journal.open('w') as f:
                for r in (config,committed,summary):f.write(json.dumps(r)+'\n')
            torch.save({'schema':'numeric_pickup_egocentric_replay_q_v2',
                'metadata':{'schema':meta['schema'],'action_names':tuple(meta['action_names'])},
                'next_episode':1,'brain':{'episodes_seen':1,'updates':12}},checkpoint)
            result=module.inspect('goodrun',mode='train',target=1,require_complete=True,verbose=False)
            self.assertEqual(result['successes'],1)
            self.assertEqual(result['optimizer_updates'],12)
            bad=dict(summary,optimizer_updates=999)
            with journal.open('w') as f:
                for r in (config,committed,bad):f.write(json.dumps(r)+'\n')
            with self.assertRaises(ValueError):module.inspect('goodrun',require_complete=True,verbose=False)
            with journal.open('w') as f:
                for r in (config,committed):f.write(json.dumps(r)+'\n')
            with self.assertRaises(ValueError):module.inspect('goodrun',require_complete=True,verbose=False)


    def test_active_status_does_not_reject_racing_atomic_checkpoint(self):
        with TemporaryDirectory() as d,patch.object(module,'ROOT',Path(d)):
            journal,checkpoint,lock=module.paths('live')
            journal.parent.mkdir();checkpoint.parent.mkdir()
            schema='numeric_pickup_egocentric_replay_q_v2'
            cfg={'stage':'config','mode':'train','metadata':{'schema':schema},
                 'frozen_source':None}
            row={'stage':'episode_committed','episode':0,'actual_acquired':True,
                 'own_inventory_gain':1,'final_held_oak':1,'scored_controller_commands':0,
                 'actor_updates':12,'item_visible_at_first_action':False}
            journal.write_text(json.dumps(cfg)+'\n'+json.dumps(row)+'\n')
            # Status took journal snapshot while later 3 episodes committed.
            torch.save({'schema':schema,'metadata':cfg['metadata'],
                'next_episode':4,'brain':{'episodes_seen':4,'updates':48}},checkpoint)
            with lock.open('a+') as held:
                fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
                result=module.inspect('live',verbose=False)
                self.assertTrue(result['active'])
                self.assertEqual(result['checkpoint_cursor'],4)
                self.assertEqual(result['episodes'],1)
                fcntl.flock(held,fcntl.LOCK_UN)
            # An orphaned checkpoint with no held live lock must be flagged.
            with self.assertRaises(ValueError):module.inspect('live',verbose=False)

    def test_empty_and_bad_path(self):
        with TemporaryDirectory() as d,patch.object(module,'ROOT',Path(d)):
            self.assertEqual(module.inspect('missing',verbose=False)['stage'],'not_started')
            with self.assertRaises(ValueError):module.inspect('../wrong',verbose=False)
            with self.assertRaises(ValueError):module.inspect('missing',require_complete=True,verbose=False)

if __name__=='__main__':unittest.main()
