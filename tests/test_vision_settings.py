import pathlib
import sys
import tempfile
import unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.vision_settings import ensure_clean_visual_options


class SettingsTests(unittest.TestCase):
    def test_replaces_duplicate_options_preserves_user_instance_fields(self):
        with tempfile.TemporaryDirectory() as d:
            path=pathlib.Path(d)/'options.txt'
            path.write_text('narrator:0\nchatVisibility:2\ntutorialStep:none\nchatVisibility:1\nchatScale:1.0\n')
            self.assertTrue(ensure_clean_visual_options(d))
            content=path.read_text()
            self.assertIn('narrator:0',content)
            self.assertIn('tutorialStep:none',content)
            self.assertEqual(content.count('chatVisibility:'),1)
            self.assertEqual(content.count('chatScale:'),1)
            self.assertIn('chatVisibility:0',content)
            self.assertIn('chatScale:0.0',content)
            self.assertFalse(ensure_clean_visual_options(d))

    def test_missing_instance_must_not_be_created(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):ensure_clean_visual_options(d)

if __name__=='__main__':unittest.main()
