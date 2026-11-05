"""End-of-world actual *held inventory* is distinct from cumulative gains."""
import pathlib,sys,unittest
from types import SimpleNamespace
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'experiments'))
from fresh_seed_treadmill import own_wood_held


class OwnWoodHeldTest(unittest.TestCase):
    def test_current_items_only_and_does_not_infer_past_pickups(self):
        def slot(id,count):return SimpleNamespace(item_id=id,count=count)
        p=SimpleNamespace(main_inventory=(slot('minecraft:oak_log',3),
                                          slot('minecraft:oak_planks',42),
                                          slot('minecraft:stone',17)),
                          offhand_inventory=(slot('minecraft:birch_log',2),))
        self.assertEqual(own_wood_held(p),5)
        p.main_inventory=()
        p.offhand_inventory=()
        self.assertEqual(own_wood_held(p),0)
        with self.assertRaises(TypeError):
            own_wood_held(object())


if __name__=='__main__':unittest.main()
