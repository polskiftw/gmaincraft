"""Numeric-only future item branch is separate from unchanged old Q data."""
import pathlib,sys,unittest
import numpy as np

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))

from gmaincraft.mcio_visible_items import (
    SCHEMA,VisibleItemsV1,VisibleDroppedItemV1)
from gmaincraft.visible_item_encoding import (
    encode_visible_items,FEATURES,MAX_ITEMS)
from gmaincraft.natural_online_policy import ID_BYTES


class ItemEncodingTests(unittest.TestCase):
    def test_camera_verified_items_not_hidden_or_legacy_network(self):
        item=VisibleDroppedItemV1('minecraft:oak_log',(10.,70.,13.),3.,(.75,.25),4)
        typed=VisibleItemsV1(SCHEMA,24.,(item,))
        enc=encode_visible_items(typed,(10.,70.,10.))
        self.assertEqual(enc.ids.shape,(MAX_ITEMS,ID_BYTES))
        self.assertEqual(enc.data.shape,(MAX_ITEMS,len(FEATURES)))
        self.assertEqual(enc.mask.tolist(),[1.]+[0.]*(MAX_ITEMS-1))
        self.assertAlmostEqual(float(enc.data[0,2]),3/24,places=5)
        self.assertAlmostEqual(float(enc.data[0,3]),.5,places=5)
        self.assertAlmostEqual(float(enc.data[0,4]),-.5,places=5)
        self.assertAlmostEqual(float(enc.data[0,-1]),.8,places=5)
        self.assertFalse(enc.mask.flags.writeable)
        self.assertFalse(enc.data.flags.writeable)
        self.assertFalse(enc.ids.flags.writeable)
        empty=encode_visible_items(VisibleItemsV1(SCHEMA,24.,()),(0.,70.,0.))
        self.assertFalse(empty.mask.any())
        self.assertFalse(empty.data.any())
        # The existing action-skill policy still has exactly the old
        # player/first-hit block fields. No new sensor silently reaches it.
        from dataclasses import fields
        from gmaincraft.natural_online_policy import NaturalObservation
        self.assertEqual(tuple(f.name for f in fields(NaturalObservation)),
                         ('own','crosshair_id','surface_ids','surface_data','surface_mask'))

    def test_guard_and_bounded_capacity(self):
        with self.assertRaises(TypeError):
            encode_visible_items(object(),(1.,2.,3.))
        with self.assertRaises(ValueError):
            encode_visible_items(VisibleItemsV1(SCHEMA,24.,()),(0.,float('nan'),0.))
        sample=VisibleDroppedItemV1('minecraft:dirt',(1.,2.,3.),4.,(.5,.5),2)
        obs=VisibleItemsV1(SCHEMA,24.,(sample,)*16)
        self.assertEqual(int(encode_visible_items(obs,(0.,0.,0.)).mask.sum()),MAX_ITEMS)


if __name__=='__main__':
    unittest.main()
