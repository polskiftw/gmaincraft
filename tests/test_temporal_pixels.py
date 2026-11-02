import pathlib
import sys
import unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from gmaincraft.temporal_pixels import temporal_encode


class TemporalRGBTests(unittest.TestCase):
    def test_temporal_encoding_batched_and_unbatched(self):
        x=np.zeros((2,6,84,84),dtype=np.uint8)
        x[:,0:3]=np.array([30,80,210],dtype=np.uint8).reshape(1,3,1,1)
        x[:,3:6]=np.array([40,60,210],dtype=np.uint8).reshape(1,3,1,1)
        encoded=temporal_encode(x,'delta')
        self.assertEqual(encoded.shape,x.shape)
        self.assertEqual(encoded.dtype,np.uint8)
        np.testing.assert_array_equal(encoded[0,:,0,0],np.array([30,80,210,138,108,128]))
        np.testing.assert_array_equal(temporal_encode(x[0],'delta'),encoded[0])
        self.assertIs(temporal_encode(x,'raw'),x)
        self.assertFalse(np.array_equal(x,encoded))
    def test_invalid_shape_and_mode(self):
        with self.assertRaises(ValueError):temporal_encode(np.zeros((2,5,84,84),dtype=np.uint8))
        with self.assertRaises(ValueError):temporal_encode(np.zeros((6,84,84),dtype=np.float32))
        with self.assertRaises(ValueError):temporal_encode(np.zeros((6,84,84),dtype=np.uint8),'hidden_pose')

if __name__=='__main__':unittest.main()
