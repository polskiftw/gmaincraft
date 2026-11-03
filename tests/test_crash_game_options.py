"""Crash-safe restoration must not mutate unfamiliar game options."""
import pathlib,sys,tempfile,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.instance_options import restore_orphaned_game_options

class CrashOptionsTests(unittest.TestCase):
    def test_verified_override_restores_original_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            original=b'narrator:0\nrenderDistance:16\n'
            (root/'options.gmaincraft_test1.backup').write_bytes(original)
            (root/'options.txt').write_bytes(b'narrator:0\nrenderDistance:4\nmaxFps:260\n')
            changed=restore_orphaned_game_options(root,
                 {'renderDistance':'4','maxFps':'260'},run_tag='test1')
            self.assertTrue(changed)
            self.assertEqual((root/'options.txt').read_bytes(),original)
            self.assertFalse((root/'options.gmaincraft_test1.backup').exists())
            self.assertFalse(restore_orphaned_game_options(root,
                 {'renderDistance':'4','maxFps':'260'},run_tag='test1'))

    def test_unexpected_user_change_fails_without_touching_backup(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            backup=root/'options.gmaincraft_test1.backup'
            backup.write_bytes(b'chatScale:0\n')
            current=root/'options.txt'
            current.write_bytes(b'chatScale:0\nuserCustomSetting:true\nmaxFps:260\n')
            with self.assertRaises(RuntimeError):
                restore_orphaned_game_options(root,{'maxFps':'260'},run_tag='test1')
            self.assertTrue(backup.exists())
            self.assertIn(b'userCustomSetting',current.read_bytes())

    def test_foreign_backup_blocks_restore(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td)
            (root/'options.gmaincraft_elsewhere.backup').write_bytes(b'xxx')
            (root/'options.txt').write_bytes(b'yyy')
            with self.assertRaises(FileExistsError):
                restore_orphaned_game_options(root,{'maxFps':'260'},run_tag='ours')

if __name__=='__main__':unittest.main()
