"""CBOR/field contract for opt-in vanilla client crosshair and own status."""
import pathlib
import sys
import unittest
from dataclasses import replace

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.mcio_status import PlayerStatusOption,PlayerStatusV1,SCHEMA
from mcio_ctrl.network import ObservationPacket


class StatusPacketTests(unittest.TestCase):
    def observation(self,**kwargs):
        payload=dict(schema=SCHEMA,hunger=17,saturation=4.2,armor_points=3,
                     air_supply=300,selected_hotbar_slot=1,dimension='minecraft:overworld',
                     targeted_block_id='minecraft:oak_log',targeted_block_pos=[3,65,8],
                     targeted_block_face='north',targeted_hit_distance=2.8)
        payload.update(kwargs)
        return ObservationPacket(options=[PlayerStatusOption(**payload)])

    def test_roundtrip_CBOR_registration_and_player_known_hit(self):
        packet=self.observation()
        decoded=ObservationPacket.unpack(packet.pack())
        result=PlayerStatusV1.from_packet(decoded)
        self.assertEqual(result.hunger,17)
        self.assertEqual(result.selected_hotbar_slot,1)
        self.assertEqual(result.crosshair_block.block_id,'minecraft:oak_log')
        self.assertEqual(result.crosshair_block.xyz,(3,65,8))
        self.assertEqual(result.crosshair_block.face,'north')
        self.assertEqual(result.dimension,'minecraft:overworld')
        self.assertEqual(tuple(vars(result)),('schema','hunger','saturation','armor_points',
                         'air_supply','selected_hotbar_slot','dimension','crosshair_block'))

    def test_miss_carries_no_world_position(self):
        p=self.observation(targeted_block_id=None,targeted_block_pos=None,
                           targeted_block_face=None,targeted_hit_distance=-1.)
        self.assertIsNone(PlayerStatusV1.from_packet(p).crosshair_block)
        with self.assertRaises(ValueError):
            PlayerStatusV1.from_packet(self.observation(targeted_block_id=None))

    def test_reject_fabricated_or_out_of_bounds(self):
        for values in ({'schema':'not_v1'}, {'hunger':-1}, {'saturation':float('nan')},
                       {'selected_hotbar_slot':18}, {'targeted_block_pos':[3,4,50000000]},
                       {'targeted_block_face':'wall'}, {'targeted_hit_distance':float('inf')},
                       {'targeted_block_id':'secret'}):
            with self.subTest(values=values),self.assertRaises(ValueError):
                PlayerStatusV1.from_packet(self.observation(**values))

    def test_requires_opt_in_packet(self):
        with self.assertRaises(RuntimeError):
            PlayerStatusV1.from_packet(ObservationPacket())

if __name__=='__main__':unittest.main()
