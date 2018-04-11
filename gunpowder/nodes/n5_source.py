import logging

from gunpowder.coordinate import Coordinate
from gunpowder.ext import z5py
from .hdf5like_source_base import Hdf5LikeSource


logger = logging.getLogger(__name__)


class N5Source(Hdf5LikeSource):
    def __get_voxel_size(self, dataset):
        try:
            logger.debug('Voxel size being reversed to account for N5 using column-major ordering')
            return Coordinate(dataset.attrs['resolution'][::-1])
        except Exception:  # todo: make specific when z5py supports it
            return None

    def __get_offset(self, dataset):
        try:
            logger.debug('Offset being reversed to account for N5 using column-major ordering')
            return Coordinate(dataset.attrs['offset'][::-1])
        except Exception:  # todo: make specific when z5py supports it
            return None

    def _open_file(self, filename):
        return z5py.File(filename, use_zarr_format=False)
