"""Read-only compact object packet, CBOR and no-fabricated-ray contract."""
import pathlib
import sys
import unittest
from dataclasses import replace

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.mcio_surface_objects import (SCHEMA,SurfaceObjectsRequest,SurfaceObjectHit,
                                            SurfaceObjectsOption,SurfaceObjectsV1)
from gmaincraft.mcio_surface_grid import (SurfaceGridOption,SurfaceRayHit,
                                         SurfaceGridV0,SCHEMA as GRID_SCHEMA)
from mcio_ctrl.network import ObservationPacket, ActionPacket


class GroupedSurfaceTests(unittest.TestCase):
    def packet(self,**changes):
        data=dict(schema=SCHEMA,columns=13,rows=9,vertical_fov_degrees=70.,
                  max_distance=64.,objects=[SurfaceObjectHit('minecraft:oak_log',
                    [2,65,4],4,3.25,15,0,[55,56])])
        data.update(changes)
        return ObservationPacket(options=[SurfaceObjectsOption(**data)])

    def test_roundtrip_and_projected_coverage(self):
        grid=SurfaceObjectsV1.from_packet(ObservationPacket.unpack(self.packet().pack()))
        self.assertEqual(grid.schema,SCHEMA)
        self.assertEqual(len(grid.objects),1)
        oak=grid.matching('minecraft:oak_log')[0]
        self.assertEqual(oak.xyz,(2,65,4))
        self.assertEqual(oak.visible_faces,('north',))
        self.assertEqual(oak.sample_fraction,2/117)
        self.assertEqual(oak.screen_center_uv,(4.0/13,4.5/9))
        self.assertEqual(grid.matching('minecraft:stone'),())
        with self.assertRaises(AttributeError):
            oak.block_id='minecraft:stone'

    def test_consistency_with_original_first_hits(self):
        grouped=self.packet().options[0]
        raw=SurfaceGridOption(GRID_SCHEMA,13,9,70.,64.,[
            SurfaceRayHit(55,'minecraft:oak_log',[2,65,4],'north',3.4,15,0),
            SurfaceRayHit(56,'minecraft:oak_log',[2,65,4],'north',3.25,15,0)])
        pkt=ObservationPacket(options=[grouped,raw])
        obj=SurfaceObjectsV1.from_packet(pkt)
        obj.assert_same_first_hits(SurfaceGridV0.from_packet(pkt))
        with self.assertRaises(AssertionError):
            bad=ObservationPacket(options=[grouped,replace(raw,hits=raw.hits[:1])])
            obj.assert_same_first_hits(SurfaceGridV0.from_packet(bad))

    def test_request_action_CBOR_roundtrip(self):
        original=ActionPacket(options=[SurfaceObjectsRequest(True)])
        decoded=ActionPacket.unpack(original.pack())
        self.assertEqual(len(decoded.options),1)
        self.assertIsInstance(decoded.options[0],SurfaceObjectsRequest)
        self.assertTrue(decoded.options[0].fresh)

    def test_fine_grid_allowed_but_unbounded_rejected(self):
        fine=SurfaceObjectsV1.from_packet(self.packet(columns=25,rows=17,objects=[]))
        self.assertEqual((fine.columns,fine.rows),(25,17))
        with self.assertRaises(ValueError):
            SurfaceObjectsV1.from_packet(self.packet(columns=31,rows=31,objects=[]))

    def test_fail_closed_and_reject_invalid(self):
        self.assertEqual(SurfaceObjectsV1.from_packet(self.packet(objects=[])).objects,())
        with self.assertRaises(RuntimeError):
            SurfaceObjectsV1.from_packet(ObservationPacket())
        base=self.packet().options[0].objects[0]
        for change in ({'schema':'v999'}, {'rows':0}, {'max_distance':600},
                       {'vertical_fov_degrees':float('nan')},
                       {'objects':[replace(base,face_mask=64)]},
                       {'objects':[replace(base,ray_indices=[55,55])]},
                       {'objects':[replace(base,ray_indices=[117])]},
                       {'objects':[base,base]},
                       {'objects':[replace(base,nearest_distance=1e30)]},
                       {'objects':[replace(base,sky_light=16)]},
                       {'objects':[replace(base,block_pos=[40000000,64,2])] },
                       {'objects':[replace(base,block_id='not_valid')]}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                SurfaceObjectsV1.from_packet(self.packet(**change))

if __name__=='__main__':unittest.main()
