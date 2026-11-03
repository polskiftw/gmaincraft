"""Fail-closed opt-in player-owned mining progress, no target leak."""
import pathlib
import sys
import unittest

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.mcio_mining_feedback import (SCHEMA,MiningFeedbackRequest,MiningFeedbackOption,
                                             MiningFeedbackV1)
from mcio_ctrl.network import ObservationPacket,ActionPacket


class MiningFeedbackTests(unittest.TestCase):
    def packet(self, **changes):
        option=dict(schema=SCHEMA,actively_destroying=True,
                    crack_stage=4,crosshair_on_block=True)
        option.update(changes)
        return ObservationPacket(options=[MiningFeedbackOption(**option)])

    def test_roundtrip(self):
        obs=ObservationPacket.unpack(self.packet().pack())
        result=MiningFeedbackV1.from_packet(obs)
        self.assertEqual(result.crack_stage,4)
        self.assertAlmostEqual(result.normalized_crack,.4)
        self.assertTrue(result.crosshair_on_block)
        self.assertEqual(tuple(vars(result)),('schema','actively_destroying',
                         'crack_stage','crosshair_on_block'))

    def test_on_demand_request_cbor_registration(self):
        pkt=ActionPacket(options=[MiningFeedbackRequest(True)])
        copy=ActionPacket.unpack(pkt.pack())
        self.assertEqual(len(copy.options),1)
        self.assertIsInstance(copy.options[0],MiningFeedbackRequest)
        self.assertTrue(copy.options[0].fresh)

    def test_live_transient_active_but_unavailable_crack_stage(self):
        # This occurred twice in 240 actual unstructured survival decisions.
        # Distinguish active input from progress; -1 means UNKNOWN stage.
        active_unknown=MiningFeedbackV1.from_packet(self.packet(
            actively_destroying=True,crack_stage=-1,crosshair_on_block=True))
        self.assertTrue(active_unknown.actively_destroying)
        self.assertEqual(active_unknown.crack_stage,-1)
        self.assertEqual(active_unknown.normalized_crack,-1.)

    def test_idle_and_no_packet(self):
        idle=MiningFeedbackV1.from_packet(self.packet(
            actively_destroying=False,crack_stage=-1,crosshair_on_block=False))
        self.assertEqual(idle.normalized_crack,-1.)
        with self.assertRaises(RuntimeError):
            MiningFeedbackV1.from_packet(ObservationPacket())

    def test_validation(self):
        invalid=({'schema':'status_v5'}, {'actively_destroying':0},
                 {'crack_stage':-3}, {'crack_stage':11},
                 {'actively_destroying':False,'crack_stage':0},
                 {'crack_stage':True}, {'crosshair_on_block':1})
        for change in invalid:
            with self.subTest(change=change),self.assertRaises(ValueError):
                MiningFeedbackV1.from_packet(self.packet(**change))

if __name__=='__main__':unittest.main()
