import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.fresh_seed_schedule import SeedSchedule,CAPACITY,SCHEMA


class FreshSeedTests(unittest.TestCase):
    def test_all_training_and_evaluation_seeds_unique_disjoint(self):
        train=SeedSchedule(917211,'train')
        evaluation=SeedSchedule(917211,'eval')
        ts={train.seed_at(i) for i in range(12000)}
        es={evaluation.seed_at(i) for i in range(12000)}
        self.assertEqual(len(ts),12000)
        self.assertEqual(len(es),12000)
        self.assertFalse(ts&es)
        self.assertTrue(4000<sum(seed%2==0 for seed in ts)<8000)
        self.assertTrue(4000<sum(seed%2!=0 for seed in es)<8000)
        self.assertEqual(train.seed_at(132),SeedSchedule(917211,'train').seed_at(132))
        self.assertNotEqual(train.seed_at(132),train.seed_at(133))
        self.assertNotEqual(train.seed_at(132),evaluation.seed_at(132))

    def test_signed_full64_and_late_cursor_no_early_collision(self):
        s=SeedSchedule(-1)
        candidates={s.seed_at(i) for i in (0,1,2,3,99999,CAPACITY-1)}
        self.assertEqual(len(candidates),6)
        self.assertTrue(all(-(1<<63)<=x<1<<63 for x in candidates))
        self.assertTrue(s.validate_seed(CAPACITY-1,s.seed_at(CAPACITY-1)))
        self.assertFalse(s.validate_seed(0,s.seed_at(1)))

    def test_invalid_seed_or_cursor_rejected(self):
        for bad in (-1,CAPACITY,True,1.3):
            with self.assertRaises(ValueError):SeedSchedule(0).seed_at(bad)
        for bad in (True,1.5,1<<63,-(1<<63)-1):
            with self.assertRaises(ValueError):SeedSchedule(bad)
        with self.assertRaises(ValueError):SeedSchedule(0,'training')

if __name__=='__main__':unittest.main()
