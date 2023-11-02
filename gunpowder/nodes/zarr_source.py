from gunpowder.ext import ZarrFile
from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.profiling import Timing
from gunpowder.roi import Roi
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from .batch_provider import BatchProvider

from zarr._storage.store import BaseStore
from zarr import N5Store, N5FSStore
import numpy as np

from collections.abc import MutableMapping
from typing import Union
import warnings
import logging


logger = logging.getLogger(__name__)


class ZarrSource(BatchProvider):
    """A `zarr <https://github.com/zarr-developers/zarr>`_ data source.

    Provides arrays from zarr datasets. If the attribute ``resolution`` is set
    in a zarr dataset, it will be used as the array's ``voxel_size``. If the
    attribute ``offset`` is set in a dataset, it will be used as the offset of
    the :class:`Roi` for this array. It is assumed that the offset is given in
    world units.

    Args:

        store (``string``, ``zarr.BaseStore``):

            A zarr store or path to a zarr directory or zip file.

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

    def __init__(
        self,
        store: Union[BaseStore, MutableMapping, str] = None,
        datasets=None,
        array_specs=None,
        channels_first=True,
        filename=None,
    ):
        # datasets is not really optional, this is for backwards compatibility
        # only
        assert datasets is not None, "Argument 'datasets' has to be provided"

        if filename is not None:
            warnings.warn(
                "Argument 'filename' will be replaced in v2.0, " "use 'store' instead",
                DeprecationWarning,
            )

            assert store is None, "If 'store' is given, 'filename' has to be None"

            store = filename

        self.store = store

        if array_specs is None:
            self.array_specs = {}
        else:
            self.array_specs = array_specs

        self.channels_first = channels_first
        self.datasets = datasets

    def _get_voxel_size(self, dataset):
        if "resolution" not in dataset.attrs:
            return None

        if self._rev_metadata():
            return Coordinate(dataset.attrs["resolution"][::-1])
        else:
            return Coordinate(dataset.attrs["resolution"])

    def _get_offset(self, dataset):
        if "offset" not in dataset.attrs:
            return None

        if self._rev_metadata():
            return Coordinate(dataset.attrs["offset"][::-1])
        else:
            return Coordinate(dataset.attrs["offset"])

    def _rev_metadata(self):
        with ZarrFile(self.store, mode="a") as store:
            return isinstance(store.chunk_store, N5Store) or isinstance(
                store.chunk_store, N5FSStore
            )

    def _open_file(self, store):
        return ZarrFile(store, mode="r")

    def setup(self):
        with self._open_file(self.store) as data_file:
            for array_key, ds_name in self.datasets.items():
                if ds_name not in data_file:
                    raise RuntimeError("%s not in %s" % (ds_name, self.store))

                spec = self.__read_spec(array_key, data_file, ds_name)

                self.provides(array_key, spec)

    def provide(self, request):
        timing = Timing(self)
        timing.start()

        batch = Batch()

        with self._open_file(self.store) as data_file:
            for array_key, request_spec in request.array_specs.items():
                logger.debug("Reading %s in %s...", array_key, request_spec.roi)

                voxel_size = self.spec[array_key].voxel_size

                # scale request roi to voxel units
                dataset_roi = request_spec.roi / voxel_size

                # shift request roi into dataset
                dataset_roi = dataset_roi - self.spec[array_key].roi.offset / voxel_size

                # create array spec
                array_spec = self.spec[array_key].copy()
                array_spec.roi = request_spec.roi

                # add array to batch
                batch.arrays[array_key] = Array(
                    self.__read(data_file, self.datasets[array_key], dataset_roi),
                    array_spec,
                )

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __read_spec(self, array_key, data_file, ds_name):
        dataset = data_file[ds_name]

        if array_key in self.array_specs:
            spec = self.array_specs[array_key].copy()
        else:
            spec = ArraySpec()

        if spec.voxel_size is None:
            voxel_size = self._get_voxel_size(dataset)
            if voxel_size is None:
                voxel_size = Coordinate((1,) * len(dataset.shape))
                logger.warning(
                    "WARNING: File %s does not contain resolution information "
                    "for %s (dataset %s), voxel size has been set to %s. This "
                    "might not be what you want.",
                    self.store,
                    array_key,
                    ds_name,
                    spec.voxel_size,
                )
            spec.voxel_size = voxel_size

        self.ndims = len(spec.voxel_size)

        if spec.roi is None:
            offset = self._get_offset(dataset)
            if offset is None:
                offset = Coordinate((0,) * self.ndims)

            if self.channels_first:
                shape = Coordinate(dataset.shape[-self.ndims :])
            else:
                shape = Coordinate(dataset.shape[: self.ndims])

            spec.roi = Roi(offset, shape * spec.voxel_size)

        if spec.dtype is not None:
            assert spec.dtype == dataset.dtype, (
                "dtype %s provided in array_specs for %s, "
                "but differs from dataset %s dtype %s"
                % (self.array_specs[array_key].dtype, array_key, ds_name, dataset.dtype)
            )
        else:
            spec.dtype = dataset.dtype

        if spec.interpolatable is None:
            spec.interpolatable = spec.dtype in [
                np.float32,
                np.float64,
                np.float128,
                np.uint8,  # assuming this is not used for labels
            ]
            logger.warning(
                "WARNING: You didn't set 'interpolatable' for %s "
                "(dataset %s). Based on the dtype %s, it has been "
                "set to %s. This might not be what you want.",
                array_key,
                ds_name,
                spec.dtype,
                spec.interpolatable,
            )

        return spec

    def __read(self, data_file, ds_name, roi):
        c = len(data_file[ds_name].shape) - self.ndims

        if self.channels_first:
            array = np.asarray(data_file[ds_name][(slice(None),) * c + roi.to_slices()])
        else:
            array = np.asarray(data_file[ds_name][roi.to_slices() + (slice(None),) * c])
            array = np.transpose(
                array, axes=[i + self.ndims for i in range(c)] + list(range(self.ndims))
            )

        return array

    def name(self):
        return super().name() + f"[{self.store}]"
