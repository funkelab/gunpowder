from collections.abc import MutableMapping
from typing import Union

from zarr._storage.store import BaseStore

from gunpowder.coordinate import Coordinate
from gunpowder.ext import ZarrFile
from .hdf5like_source_base import Hdf5LikeSource


class ZarrSource(Hdf5LikeSource):
    """A `zarr <https://github.com/zarr-developers/zarr>`_ data source.

    Provides arrays from zarr datasets. If the attribute ``resolution`` is set
    in a zarr dataset, it will be used as the array's ``voxel_size``. If the
    attribute ``offset`` is set in a dataset, it will be used as the offset of
    the :class:`Roi` for this array. It is assumed that the offset is given in
    world units.

    Args:

        store (``string or ZarrStore``):

            The zarr directory.

        datasets (``dict``, :class:`ArrayKey` -> ``string``):

            Dictionary of array keys to dataset names that this source offers.

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`, optional):

            An optional dictionary of array keys to array specs to overwrite
            the array specs automatically determined from the data file. This
            is useful to set a missing ``voxel_size``, for example. Only fields
            that are not ``None`` in the given :class:`ArraySpec` will be used.

        channels_first (``bool``, optional):

            Specifies the ordering of the dimensions of the HDF5-like data source.
            If channels_first is set (default), then the input shape is expected
            to be (channels, spatial dimensions). This is recommended because of
            better performance. If channels_first is set to false, then the input
            data is read in channels_last manner and converted to channels_first.
    """

    def __init__(self, store: Union[BaseStore, MutableMapping, str], datasets, array_specs=None, channels_first=True):
        super().__init__(store, datasets, array_specs, channels_first)

    def _get_voxel_size(self, dataset):

        if 'resolution' not in dataset.attrs:
            return None

        if self.filename.endswith('.n5'):
            return Coordinate(dataset.attrs['resolution'][::-1])
        else:
            return Coordinate(dataset.attrs['resolution'])

    def _get_offset(self, dataset):

        if 'offset' not in dataset.attrs:
            return None
        if isinstance(self.filename, str):
            if self.filename.endswith('.n5'):
                return Coordinate(dataset.attrs['offset'][::-1])
        else:
            return Coordinate(dataset.attrs['offset'])

    def _open_file(self, store):
        return ZarrFile(store, mode='r')
