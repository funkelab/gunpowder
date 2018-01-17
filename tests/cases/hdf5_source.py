from .provider_test import ProviderTest
from gunpowder import *
import numpy as np
from gunpowder.ext import h5py

class TestHdf5Source(ProviderTest):

    def test_output_2d(self):

        # create a test file
        with h5py.File('test_hdf_source.hdf', 'w') as f:
            f['raw'] = np.zeros((100, 100), dtype=np.float32)
            f['raw_low'] = np.zeros((10, 10), dtype=np.float32)
            f['raw_low'].attrs['resolution'] = (10, 10)
            f['seg'] = np.ones((100, 100), dtype=np.uint64)

        # read arrays
        raw = ArrayKey('RAW')
        raw_low = ArrayKey('RAW_LOW')
        seg = ArrayKey('SEG')
        source = Hdf5Source(
            'test_hdf_source.hdf',
            {
                raw: 'raw',
                raw_low: 'raw_low',
                seg: 'seg'
            }
        )

        with build(source):

            batch = source.request_batch(
                BatchRequest({
                    raw: ArraySpec(roi=Roi((0, 0), (100, 100))),
                    raw_low: ArraySpec(roi=Roi((0, 0), (100, 100))),
                    seg: ArraySpec(roi=Roi((0, 0), (100, 100))),
                })
            )

            self.assertTrue(batch.arrays[raw].spec.interpolatable)
            self.assertTrue(batch.arrays[raw_low].spec.interpolatable)
            self.assertFalse(batch.arrays[seg].spec.interpolatable)

    def test_output_3d(self):

        # create a test file
        with h5py.File('test_hdf_source.hdf', 'w') as f:
            f['raw'] = np.zeros((100, 100, 100), dtype=np.float32)
            f['raw_low'] = np.zeros((10, 10, 10), dtype=np.float32)
            f['raw_low'].attrs['resolution'] = (10, 10, 10)
            f['seg'] = np.ones((100, 100, 100), dtype=np.uint64)

        # read arrays
        raw = ArrayKey('RAW')
        raw_low = ArrayKey('RAW_LOW')
        seg = ArrayKey('SEG')
        source = Hdf5Source(
            'test_hdf_source.hdf',
            {
                raw: 'raw',
                raw_low: 'raw_low',
                seg: 'seg'
            }
        )

        with build(source):

            batch = source.request_batch(
                BatchRequest({
                    raw: ArraySpec(roi=Roi((0, 0, 0), (100, 100, 100))),
                    raw_low: ArraySpec(roi=Roi((0, 0, 0), (100, 100, 100))),
                    seg: ArraySpec(roi=Roi((0, 0, 0), (100, 100, 100))),
                })
            )

            self.assertTrue(batch.arrays[raw].spec.interpolatable)
            self.assertTrue(batch.arrays[raw_low].spec.interpolatable)
            self.assertFalse(batch.arrays[seg].spec.interpolatable)
