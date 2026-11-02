"""Policy input/action contracts do not require a Minecraft installation."""
import pathlib
import sys
import unittest
from dataclasses import fields
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from gmaincraft.agent_interface import AgentAction, AgentView, natural_log_count


class FakePacket:
    # Deliberately include secrets a learner must not receive.
    player_pos = (999, 55, 888)
    player_yaw = 123
    world_seed = 98765
    inventory_main = [SimpleNamespace(id='minecraft:oak_log', count=2),
                      SimpleNamespace(id='minecraft:cherry_log', count=1),
                      SimpleNamespace(id='minecraft:oak_planks', count=42)]
    def get_frame(self):
        return self.frame

    def __init__(self):
        self.frame = np.zeros((100, 200, 3), dtype=np.uint8)


class InterfaceTest(unittest.TestCase):
    def test_pixel_only_boundary_and_defensive_copy(self):
        pkt = FakePacket()
        view = AgentView.from_packet(pkt)
        self.assertEqual(tuple(f.name for f in fields(view)), ('pixels',))
        self.assertEqual(view.pixels.shape, (100, 200, 3))
        self.assertFalse(view.pixels.flags.writeable)
        pkt.frame[0, 0, 0] = 99
        self.assertEqual(int(view.pixels[0, 0, 0]), 0)
        self.assertFalse(hasattr(view, 'player_pos'))
        self.assertFalse(hasattr(view, 'inventory_main'))
        self.assertFalse(hasattr(view, 'world_seed'))

    def test_action_rejects_cheat_and_invalid_inputs(self):
        self.assertEqual(AgentAction(keys=('W', 'LEFT_BUTTON'), ticks=100).ticks, 100)
        for kwargs in ({'keys': ('/tp',)}, {'keys': ('W', 'W')},
                       {'ticks': 0}, {'ticks': 201}, {'ticks': True},
                       {'look_dx': float('nan')}, {'look_dy': float('inf')},
                       {'look_dx': 1300}, {'keys': ['W']}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AgentAction(**kwargs)
        self.assertNotIn('commands', (f.name for f in fields(AgentAction)))

    def test_natural_log_scoring_is_distinct_from_agent_input(self):
        self.assertEqual(natural_log_count(FakePacket()), 3)
        self.assertEqual(natural_log_count(SimpleNamespace(inventory_main=[])), 0)

    def test_non_rgb_packet_rejected(self):
        p = FakePacket()
        p.frame = np.zeros((20, 30), dtype=np.uint8)
        with self.assertRaises(ValueError):
            AgentView.from_packet(p)


if __name__ == '__main__':
    unittest.main()
