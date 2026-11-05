"""Pure, no-JVM safety rules for immutable same-world snapshot comparisons."""
import pathlib,sys,tempfile,unittest
from unittest import mock
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments'))
from gmaincraft.fresh_world_manager import FreshWorldManager,names
from fresh_seed_treadmill import parse_args,metadata,paired_frozen_q_sampler_seed


class PairedSourceTests(unittest.TestCase):
    def test_shared_source_file_kept_across_both_separate_eval_arms(self):
        with tempfile.TemporaryDirectory() as d, mock.patch(
            'gmaincraft.fresh_world_manager.world.WorldManager') as mclass, mock.patch(
            'gmaincraft.fresh_world_manager.verify_noise_world') as verify, mock.patch(
            'gmaincraft.fresh_world_manager.fix_spawn_radius') as fix:
            base=pathlib.Path(d)
            source=base/'world_storage'/names('setcache',31)[0]
            source.mkdir(parents=True)
            def copy(source_kind,from_name,instance,to_name,*,force):
                self.assertEqual(source_kind,'storage')
                self.assertEqual(from_name,names('setcache',31)[0])
                (base/'instances'/instance/'saves'/to_name).mkdir(parents=True)
            mclass.return_value.copy.side_effect=copy
            for tag in ('evalpair_s','evalpair_p'):
                manager=FreshWorldManager(base,tag,paired_source_tag='setcache')
                self.assertEqual(manager.paths(31)[0],source)
                name=manager.prepare(31,123456)
                self.assertEqual(name,names(tag,31)[1])
                mclass.return_value.create.assert_not_called()
                self.assertTrue(source.is_dir())
                manager.retire(31)
                self.assertTrue(source.is_dir())
                manager.quarantine(31)
                self.assertTrue(source.is_dir())
            self.assertEqual(mclass.return_value.copy.call_count,2)
            self.assertGreaterEqual(verify.call_count,4)
            fix.assert_called()

    def test_frozen_q_episode_sampler_is_stable_and_separate_between_worlds(self):
        args=parse_args(['--run-tag','paired_s','--mode','eval',
            '--control','q_sampled','--q-model','qv2memf1104',
            '--q-goal','effect:visible_wood_crack_progress_v2',
            '--reward-profile','wood_temporal_v9','--pickup-model','hardrep1105b',
            '--paired-world-cache','sharedsrc'])
        a=paired_frozen_q_sampler_seed(args,2000)
        self.assertEqual(a,paired_frozen_q_sampler_seed(args,2000))
        self.assertNotEqual(a,paired_frozen_q_sampler_seed(args,2001))
        self.assertEqual(metadata(args)['paired_q_rng_per_world'],
                         'sha256_policy_root_episode_v1')
        args.paired_world_cache=None
        with self.assertRaises(ValueError):paired_frozen_q_sampler_seed(args,2000)

    def test_shared_cache_validation_and_v9_eval_only(self):
        with tempfile.TemporaryDirectory() as d, mock.patch(
            'gmaincraft.fresh_world_manager.world.WorldManager'):
            with self.assertRaises(ValueError):
                FreshWorldManager(d,'runx',paired_source_tag='runx')
            with self.assertRaises(ValueError):
                FreshWorldManager(d,'runx',paired_source_tag='../invalid')
        flags=['--run-tag','paired_s','--mode','eval','--control','q_sampled',
               '--q-model','qv2memf1104','--q-goal','effect:visible_wood_crack_progress_v2',
               '--reward-profile','wood_temporal_v9',
               '--pickup-model','hardrep1105b','--paired-world-cache','sharedsrc',
               '--pickup-selector','shadow_budget_v2']
        args=parse_args(flags)
        self.assertEqual(metadata(args)['paired_world_cache'],'sharedsrc')
        with self.assertRaises(SystemExit):
            parse_args(['--run-tag','training','--paired-world-cache','sharedsrc'])

if __name__=='__main__':unittest.main()
