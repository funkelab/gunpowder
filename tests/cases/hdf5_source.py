from unittest import skipIf

from .provider_test import ProviderTest
from gunpowder import *
import numpy as np
from gunpowder.ext import h5py, zarr, ZarrFile, z5py, NoSuchModule


class Hdf5LikeSourceTestMixin(object):
    """This class is to be used as a mixin for ProviderTest classes testing HDF5, N5 and Zarr
    batch providers.

    Subclasses must define ``extension`` and ``SourceUnderTest`` class variables, and an
    ``_open_writable_file(self, path)`` method. See TestHdf5Source for examples.
    """
    extension = None
    SourceUnderTest = None

    def _open_writable_file(self, path):
        raise NotImplementedError('_open_writable_file should be overridden')

    def _create_dataset(self, data_file, key, data, chunks=None, **kwargs):
        chunks = chunks or data.shape
        d = data_file.create_dataset(key, shape=data.shape, dtype=data.dtype, chunks=chunks)
        d[:] = data
        for key, value in kwargs.items():
            d.attrs[key] = value

    def test_output_2d(self):
        path = self.path_to('test_{0}_source.{0}'.format(self.extension))

        with self._open_writable_file(path) as f:
            self._create_dataset(f, 'raw', np.zeros((100, 100), dtype=np.float32))
            self._create_dataset(f, 'raw_low', np.zeros((10, 10), dtype=np.float32), resolution=(10, 10))
            self._create_dataset(f, 'seg', np.ones((100, 100), dtype=np.uint64))

        # read arrays
        raw = ArrayKey('RAW')
        raw_low = ArrayKey('RAW_LOW')
        seg = ArrayKey('SEG')
        source = self.SourceUnderTest(
            path,
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
        path = self.path_to('test_{0}_source.{0}'.format(self.extension))

        # create a test file
        with self._open_writable_file(path) as f:
            self._create_dataset(f, 'raw', np.zeros((100, 100, 100), dtype=np.float32))
            self._create_dataset(f, 'raw_low', np.zeros((10, 10, 10), dtype=np.float32), resolution=(10, 10, 10))
            self._create_dataset(f, 'seg', np.ones((100, 100, 100), dtype=np.uint64))

        # read arrays
        raw = ArrayKey('RAW')
        raw_low = ArrayKey('RAW_LOW')
        seg = ArrayKey('SEG')
        source = self.SourceUnderTest(
            path,
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


class TestHdf5Source(ProviderTest, Hdf5LikeSourceTestMixin):
    extension = 'hdf'
    SourceUnderTest = Hdf5Source

    def _open_writable_file(self, path):
        return h5py.File(path, 'w')


@skipIf(isinstance(zarr, NoSuchModule), 'zarr is not installed')
class TestZarrSource(ProviderTest, Hdf5LikeSourceTestMixin):
    extension = 'zarr'
    SourceUnderTest = ZarrSource

    def _open_writable_file(self, path):
        return ZarrFile(path, mode='w')


@skipIf(isinstance(z5py, NoSuchModule), 'z5py is not installed')
class TestN5Source(ProviderTest, Hdf5LikeSourceTestMixin):
    extension = 'n5'
    SourceUnderTest = N5Source

    def _open_writable_file(self, path):
        return z5py.File(path, use_zarr_format=False)
