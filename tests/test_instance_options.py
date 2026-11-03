"""Options A/B experiments must never damage the original local instance."""
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.instance_options import temporary_game_options


class OptionsTests(unittest.TestCase):
    def test_temporary_changes_restore_exact_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory)
            config=root/'options.txt'
            data=b'chatScale:0.0\nmaxFps:60\nunknownSetting:example\n'
            config.write_bytes(data)
            with temporary_game_options(root,dict(maxFps='260',enableVsync='false'),run_tag='test01'):
                text=config.read_text()
                self.assertIn('maxFps:260\n',text)
                self.assertIn('enableVsync:false\n',text)
                self.assertNotIn('maxFps:60\n',text)
                self.assertTrue((root/'options.gmaincraft_test01.backup').exists())
            self.assertEqual(config.read_bytes(),data)
            self.assertFalse((root/'options.gmaincraft_test01.backup').exists())

    def test_exception_also_restores_and_leftover_backup_blocks_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory)
            config=root/'options.txt'
            config.write_bytes(b'chatScale:0.0\n')
            with self.assertRaises(ArithmeticError):
                with temporary_game_options(root,{'maxFps':'260'},run_tag='test02'):
                    raise ArithmeticError('intentional')
            self.assertEqual(config.read_bytes(),b'chatScale:0.0\n')
            (root/'options.gmaincraft_old.backup').write_text('old')
            with self.assertRaises(FileExistsError):
                with temporary_game_options(root,{'maxFps':'260'},run_tag='test03'):
                    pass
            self.assertEqual(config.read_bytes(),b'chatScale:0.0\n')
            with self.assertRaises(ValueError):
                with temporary_game_options(root,{'maxFps':'\nmalicious'},run_tag='valid'):
                    pass

if __name__=='__main__': unittest.main()
