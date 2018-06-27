import logging

from gunpowder.coordinate import Coordinate
from gunpowder.ext import z5py
from .hdf5like_source_base import Hdf5LikeSource


logger = logging.getLogger(__name__)


class N5Source(Hdf5LikeSource):
    '''An `N5 <https://github.com/saalfeldlab/n5>`_ data source.

    Provides arrays from N5 datasets. If the attribute ``resolution`` is set
    in a N5 dataset, it will be used as the array's ``voxel_size``. If the
    attribute ``offset`` is set in a dataset, it will be used as the offset of
    the :class:`Roi` for this array. It is assumed that the offset is given in
    world units.

    Args:

        filename (``string``):

            The N5 directory.

        datasets (``dict``, :class:`ArrayKey` -> ``string``):

            Dictionary of array keys to dataset names that this source offers.

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`, optional):

            An optional dictionary of array keys to array specs to overwrite
            the array specs automatically determined from the data file. This
            is useful to set a missing ``voxel_size``, for example. Only fields
            that are not ``None`` in the given :class:`ArraySpec` will be used.
    '''
    def _get_voxel_size(self, dataset):
        try:
            logger.debug('Voxel size being reversed to account for N5 using column-major ordering')
            return Coordinate(dataset.attrs['resolution'][::-1])
        except Exception:  # todo: make specific when z5py supports it
            return None

    def _get_offset(self, dataset):
        try:
            logger.debug('Offset being reversed to account for N5 using column-major ordering')
            return Coordinate(dataset.attrs['offset'][::-1])
        except Exception:  # todo: make specific when z5py supports it
            return None

    def _open_file(self, filename):
        return z5py.File(filename, use_zarr_format=False)
