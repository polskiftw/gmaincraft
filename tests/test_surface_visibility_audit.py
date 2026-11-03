"""Pure negative/positive controls for independent rendered-pixel evidence."""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.surface_visibility_audit import compare_frames


class FrameAuditTests(unittest.TestCase):
    def test_identical_and_tiny_noise(self):
        a=np.zeros((180,320,3),dtype=np.uint8)
        b=a.copy()
        self.assertEqual(compare_frames(a,b).changed_pixels,0)
        b[50:70,90:110,:]=20
        self.assertEqual(compare_frames(a,b).changed_pixels,0)

    def test_visible_change_and_hud_crosshair_exclusion(self):
        a=np.zeros((180,320,3),dtype=np.uint8)
        b=a.copy()
        b[55:64,74:84,:]=250
        b[160:170,60:90,:]=255  # HUD must be excluded
        b[89:93,159:163,:]=255  # Crosshair must be excluded
        p=compare_frames(a,b)
        self.assertEqual(p.changed_pixels,90)
        self.assertEqual(p.bbox_xyxy,(74,55,84,64))
        self.assertEqual(p.max_channel_delta,250)

    def test_bad_frames_rejected(self):
        a=np.zeros((180,320,3),dtype=np.uint8)
        for b in (a[:100],a.astype(np.float32),a[:,:,0]):
            with self.assertRaises(ValueError):compare_frames(a,b)

if __name__=='__main__':unittest.main()
