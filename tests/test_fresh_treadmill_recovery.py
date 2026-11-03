"""No Minecraft launches: rollback, atomic checkpoint and input contracts."""
import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments'))
from gmaincraft.fresh_seed_schedule import SeedSchedule
from gmaincraft.fresh_world_manager import FreshWorldManager,names
from gmaincraft.natural_online_policy import NaturalPPO
from fresh_seed_treadmill import (parse_args,metadata,save_json_atomic,
    save_model,restore,recover_inflight,repair_jsonl_tail,quarantine_pending_file)


class StreamRecoveryTests(unittest.TestCase):
    def test_world_names_are_owned_and_long_safe(self):
        self.assertEqual(names('run1',3)[0],
                         'MDL_FreshStream_run1_000000000003S')
        self.assertTrue(names('safe_tag',20)[1].endswith('000000000020W'))
        for tag in ('../escape','ab/c','a'*40,'a b'):
            with self.assertRaises(ValueError):names(tag,0)
        with self.assertRaises(ValueError):names('tag',-1)

    def test_dry_run_cli_does_not_require_checkpoint_or_world(self):
        args=parse_args(['--run-tag','tmpstream','--decisions','32',
                         '--episode-decisions','12','--dry-run'])
        self.assertEqual(args.mode,'train')
        self.assertEqual(args.decisions,32)
        self.assertEqual(metadata(args)['seed_schedule'],
            'minecraft_seed_stream_v2_full64_interleaved_train_eval')
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(['--run-tag','tmpstream','--mode','eval','--control','policy_sampled',
                            '--decisions','10'])
            with self.assertRaises(SystemExit):
                parse_args(['--run-tag','../x'])

    def test_checkpoint_tied_to_exact_seed_stream_and_lossless_cursor(self):
        args=parse_args(['--run-tag','safe1','--episode-decisions','12',
                         '--update-every','8','--seed-root','2121'])
        meta=metadata(args)
        with tempfile.TemporaryDirectory() as td:
            file=pathlib.Path(td)/'t.pt'
            actor=NaturalPPO(seed=2133,device='cpu')
            save_model(file,actor,meta,13,445,{'fresh_cell':12})
            newer=NaturalPPO(seed=5,device='cpu')
            cursor,decisions,totals=restore(file,newer,meta)
            self.assertEqual((cursor,decisions,totals['fresh_cell']),(13,445,12))
            self.assertEqual(actor.state()['updates'],newer.state()['updates'])
            import torch
            for key,value in actor.state()['net'].items():
                self.assertTrue(torch.equal(value,newer.state()['net'][key]))
            changed=dict(meta,episode_decisions=13)
            with self.assertRaises(ValueError):restore(file,newer,changed)

    def test_atomic_marker_and_crash_quarantine_not_fake_progress(self):
        with tempfile.TemporaryDirectory() as td:
            marker=pathlib.Path(td)/'inflight.json'
            run=SeedSchedule(123,'train')
            save_json_atomic(marker,dict(episode=5,seed=run.seed_at(5),
                                         seed_root=123,namespace='train'))
            manager=mock.Mock()
            manager.quarantine.return_value=['/safe/private/abandoned_save']
            log=io.StringIO()
            recover_inflight(marker,5,manager,log)
            self.assertFalse(marker.exists())
            manager.quarantine.assert_called_once_with(5)
            self.assertEqual(json.loads(log.getvalue())['stage'],'recovered_aborted_episode')
            self.assertNotIn('learned',log.getvalue())
            save_json_atomic(marker,dict(episode=5,seed=run.seed_at(5),
                                         seed_root=123,namespace='train'))
            log=io.StringIO()
            recover_inflight(marker,6,manager,log)
            manager.retire.assert_called_once_with(5,keep=False)
            self.assertEqual(json.loads(log.getvalue())['stage'],'recovered_stale_marker')

    def test_crashed_partial_jsonl_tail_preserved_not_credited(self):
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td)/'run.jsonl'
            p.write_bytes(b'{"stage":"config"}\n{"stage":"uncommitted_update","episode":')
            repair=repair_jsonl_tail(p)
            self.assertEqual(repair['stage'],'recovered_partial_journal_line')
            self.assertEqual(p.read_text(),'{'+'"stage":"config"}'+'\n')
            self.assertTrue(pathlib.Path(repair['quarantined']).read_bytes().startswith(
                            b'{"stage":"uncommitted_update"'))
            self.assertIsNone(repair_jsonl_tail(p))
            p.write_bytes(b'{"stage":"config"}\n{"stage":"completed"}')
            result=repair_jsonl_tail(p)
            self.assertEqual(result['stage'],'recovered_missing_journal_newline')
            self.assertEqual(len(p.read_text().splitlines()),2)

    def test_stale_pending_checkpoint_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            staged=pathlib.Path(td)/'model.pt.pending'
            staged.write_bytes(b'possibly truncated old CUDA model')
            moved=quarantine_pending_file(staged)
            self.assertFalse(staged.exists())
            self.assertEqual(pathlib.Path(moved).read_bytes(),b'possibly truncated old CUDA model')
            self.assertIsNone(quarantine_pending_file(staged))

    def test_corrupt_markers_do_not_change_any_world(self):
        with tempfile.TemporaryDirectory() as td:
            marker=pathlib.Path(td)/'marker'
            manager=mock.Mock()
            save_json_atomic(marker,dict(episode=4,seed=0,
                                         seed_root=123,namespace='train'))
            with self.assertRaises(ValueError):recover_inflight(marker,4,manager,io.StringIO())
            self.assertTrue(marker.exists())
            manager.quarantine.assert_not_called()
            manager.retire.assert_not_called()

if __name__=='__main__':unittest.main()
