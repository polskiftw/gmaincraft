"""Schema, CBOR, immutability and fail-closed checks of research-only v0 rays."""
import pathlib
import sys
import unittest
from dataclasses import replace

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.mcio_surface_grid import (SCHEMA, SurfaceRayHit, SurfaceGridOption,
                                         SurfaceGridV0)
from mcio_ctrl.network import ObservationPacket


class SurfaceGridTests(unittest.TestCase):
    def packet(self, **changes):
        data=dict(schema=SCHEMA,columns=13,rows=9,vertical_fov_degrees=70.,
                  max_distance=64.,hits=[SurfaceRayHit(58,'minecraft:oak_log',
                    [1,65,4],'north',3.,15,0)])
        data.update(changes)
        return ObservationPacket(options=[SurfaceGridOption(**data)])

    def test_roundtrip_and_immutability(self):
        packet=ObservationPacket.unpack(self.packet().pack())
        grid=SurfaceGridV0.from_packet(packet)
        self.assertEqual(len(grid.hits),1)
        self.assertEqual(grid.hits[0].xyz,(1,65,4))
        self.assertEqual(grid.hits[0].normalized_uv,(.5,.5))
        self.assertEqual(grid.block_hits('minecraft:oak_log'),grid.hits)
        with self.assertRaises(AttributeError):
            grid.hits[0].block_id='minecraft:diamond_ore'
        self.assertEqual(grid.block_hits('minecraft:stone'),())

    def test_empty_hits_carry_no_world_information(self):
        self.assertEqual(SurfaceGridV0.from_packet(self.packet(hits=[])).hits,())
        with self.assertRaises(RuntimeError):
            SurfaceGridV0.from_packet(ObservationPacket())

    def test_reject_bad_inputs(self):
        hit=SurfaceRayHit(58,'minecraft:oak_log',[1,65,4],'north',3.,15,0)
        for change in ({'schema':'v5'}, {'columns':0}, {'rows':500},
                       {'vertical_fov_degrees':float('nan')}, {'max_distance':600},
                       {'hits':[replace(hit,ray_index=117)]},
                       {'hits':[hit,hit]}, {'hits':[replace(hit,block_pos=[1,65,31_000_000])]},
                       {'hits':[replace(hit,face='inside')]},
                       {'hits':[replace(hit,distance=65)]},
                       {'hits':[replace(hit,block_light=16)]},
                       {'hits':[replace(hit,block_id='hidden_ore')]}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                SurfaceGridV0.from_packet(self.packet(**change))

if __name__=='__main__':unittest.main()
