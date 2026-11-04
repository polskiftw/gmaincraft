"""Memory policy actions must not decode pixels; legacy RGB remains default."""
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np
from mcio_ctrl import network

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.agent_interface import AgentAction,AgentView,MCioAgentBridge


class CountingPacket:
    version=network.MCIO_PROTOCOL_VERSION
    cursor_pos=(10.,20.)

    def __init__(self):
        self.calls=0

    def get_frame(self):
        self.calls+=1
        return np.zeros((4,8,3),dtype=np.uint8)


class MemoryOnlyBridgeTest(unittest.TestCase):
    def setUp(self):
        self.packet=CountingPacket()
        self.sent=[]
        self.bridge=object.__new__(MCioAgentBridge)
        self.bridge.packet=self.packet
        self.bridge.input_map={'W':1,'LEFT_BUTTON':2}
        self.bridge.input_manager=SimpleNamespace(process_action=lambda held,mapping: [])
        self.bridge._stats=SimpleNamespace(update_cache=lambda packet: None)
        self.bridge.controller=SimpleNamespace(
            send_action=lambda action:self.sent.append(action),
            recv_observation=lambda:self.packet)

    def test_memory_only_action_advances_without_rgb(self):
        result=self.bridge.apply(AgentAction(keys=('W',),look_dx=5.,ticks=3),
                                 return_view=False)
        self.assertIsNone(result)
        self.assertEqual(self.packet.calls,0)
        self.assertEqual(len(self.sent),3)
        self.assertEqual(self.sent[0].cursor_pos,[(15.,20.)])
        self.assertFalse(self.sent[1].cursor_pos)

    def test_legacy_action_still_returns_defensive_copy(self):
        view=self.bridge.apply(AgentAction(ticks=1))
        self.assertIsInstance(view,AgentView)
        self.assertEqual(self.packet.calls,1)
        self.assertEqual(view.pixels.shape,(4,8,3))
        self.assertFalse(view.pixels.flags.writeable)


if __name__=='__main__':
    unittest.main()
