"""Pure log-curriculum invariants; no MCio or Minecraft needed."""
from dataclasses import dataclass
import pathlib
import sys
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from gmaincraft.log_curriculum import validate_tag, inventory_count, relative_cursor


@dataclass
class Slot:
    id: str
    count: int


@dataclass
class FakeObservation:
    inventory_main: list[Slot]
    cursor_pos: tuple[float,float] = (10.0, 20.0)


class LogCurriculumTests(unittest.TestCase):
    def test_tag_keeps_disposable_prefix(self):
        self.assertEqual(validate_tag('test-7'), 'MDL_LogInputProbe_test-7')

    def test_rejects_path_traversal_and_bad_tags(self):
        for tag in ('', '../real', '/tmp', 'a/b', '.hidden', 'a'*41):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                validate_tag(tag)

    def test_inventory_count_requires_observed_positive_oak(self):
        ob = FakeObservation([Slot('minecraft:oak_log',2),Slot('minecraft:oak_log',1),
                              Slot('minecraft:birch_log',2), Slot('minecraft:oak_log',0)])
        self.assertEqual(inventory_count(ob),3)
        self.assertEqual(inventory_count(FakeObservation([])),0)

    def test_camera_delta_is_absolute_packet_cursor(self):
        ob=FakeObservation([])
        self.assertEqual(relative_cursor(ob,120,-50), [(130.0,-30.0)])
        self.assertEqual(ob.cursor_pos, (10.0,20.0))


if __name__ == '__main__':
    unittest.main()
