import logging
import numpy as np

from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.profiling import Timing
from gunpowder.roi import Roi
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from .batch_provider import BatchProvider

from typing import List

logger = logging.getLogger(__name__)


class Hdf5LikeSource(BatchProvider):
    '''An HDF5-like data source.

    Provides arrays from datasets accessed with an h5py-like API for each array
    key given. If the attribute ``resolution`` is set in a dataset, it will be
    used as the array's ``voxel_size``. If the attribute ``offset`` is set in a
    dataset, it will be used as the offset of the :class:`Roi` for this array.
    It is assumed that the offset is given in world units.

    Args:

        filename (``string``):

            The input file.

        datasets (``dict``, :class:`ArrayKey` -> ``string``):

            Dictionary of array keys to dataset names that this source offers.

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`, optional):

            An optional dictionary of array keys to array specs to overwrite
            the array specs automatically determined from the data file. This
            is useful to set a missing ``voxel_size``, for example. Only fields
            that are not ``None`` in the given :class:`ArraySpec` will be used.

        transpose (``list``, optional):

            Specifies the order in which to permute the axes. Given a tensor
            of the form xyzc, use transpose=[3,2,1,0] to use the data as if it
            were in czyx ordering.
    '''
    def __init__(
            self,
            filename,
            datasets,
            array_specs=None,
            transpose=None):

        self.filename = filename
        self.datasets = datasets

        if array_specs is None:
            self.array_specs = {}
        else:
            self.array_specs = array_specs

        self.transpose = transpose

        # number of spatial dimensions
        self.ndims = None

    def _open_file(self, filename):
        raise NotImplementedError('Only implemented in subclasses')

    def setup(self):
        with self._open_file(self.filename) as data_file:
            for (array_key, ds_name) in self.datasets.items():

                if ds_name not in data_file:
                    raise RuntimeError("%s not in %s" % (ds_name, self.filename))

                spec = self.__read_spec(array_key, data_file, ds_name)

                self.provides(array_key, spec)

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

        with self._open_file(self.filename) as data_file:
            for (array_key, request_spec) in request.array_specs.items():
                logger.debug("Reading %s in %s...", array_key, request_spec.roi)

                voxel_size = self.spec[array_key].voxel_size

                # scale request roi to voxel units
                dataset_roi = request_spec.roi / voxel_size

                # permute roi by the inverse of the desired transpose
                # this allows for slicing then transposing vs transposing then slicing
                if self.transpose is not None:
                    dataset_roi.permute(
                        self._invert_permutation(self.transpose, self.ndims)
                    )

                # shift request roi into dataset
                dataset_roi = dataset_roi - self.spec[array_key].roi.get_offset() / voxel_size

                # create array spec
                array_spec = self.spec[array_key].copy()
                array_spec.roi = request_spec.roi

                # read array
                requested_data = self.__read(
                    data_file, self.datasets[array_key], dataset_roi
                )
                # transpose data
                if self.transpose is not None:
                    requested_data = requested_data.transpose(self.transpose)

                # add array to batch
                batch.arrays[array_key] = Array(requested_data, array_spec)

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def _get_voxel_size(self, dataset):
        try:
            return Coordinate(dataset.attrs['resolution'])
        except Exception:  # todo: make specific when z5py supports it
            return None

    def _get_offset(self, dataset):
        try:
            return Coordinate(dataset.attrs['offset'])
        except Exception:  # todo: make specific when z5py supports it
            return None

    def _invert_permutation(self, permutation: List[int], ndim: int):
        assert set(permutation) == set(
            range(ndim)
        ), "{} is an invalid permutation for a vector of length {}".format(
            permutation, ndim
        )
        inv_transpose = [0] * (ndim)
        for i in range(0, ndim):
            inv_transpose[permutation[i]] = i
        return tuple(inv_transpose)

    def __read_spec(self, array_key, data_file, ds_name):

        dataset = data_file[ds_name]

        if array_key in self.array_specs:
            spec = self.array_specs[array_key].copy()
        else:
            spec = ArraySpec()

        if spec.voxel_size is None:
            voxel_size = self._get_voxel_size(dataset)
            if voxel_size is None:
                voxel_size = Coordinate((1,)*len(dataset.shape))
                logger.warning("WARNING: File %s does not contain resolution information "
                               "for %s (dataset %s), voxel size has been set to %s. This "
                               "might not be what you want.",
                               self.filename, array_key, ds_name, spec.voxel_size)
            spec.voxel_size = voxel_size

        self.ndims = len(spec.voxel_size)

        if spec.roi is None:
            # Leave roi in original coordinate system
            offset = self._get_offset(dataset)
            if offset is None:
                offset = Coordinate((0,)*self.ndims)

            # What happens if self.ndims < len(dataset.shape)? Is this allowed? 
            # it will probably ruin the transpose functionality    
            shape = Coordinate(dataset.shape[: self.ndims])
            spec.roi = Roi(offset, shape * spec.voxel_size)

        if spec.dtype is not None:
            assert spec.dtype == dataset.dtype, ("dtype %s provided in array_specs for %s, "
                                                 "but differs from dataset %s dtype %s" %
                                                 (self.array_specs[array_key].dtype,
                                                  array_key, ds_name, dataset.dtype))
        else:
            spec.dtype = dataset.dtype

        if spec.interpolatable is None:
            spec.interpolatable = spec.dtype in [
                np.float,
                np.float32,
                np.float64,
                np.float128,
                np.uint8  # assuming this is not used for labels
            ]
            logger.warning("WARNING: You didn't set 'interpolatable' for %s "
                           "(dataset %s). Based on the dtype %s, it has been "
                           "set to %s. This might not be what you want.",
                           array_key, ds_name, spec.dtype,
                           spec.interpolatable)

        return spec

    def __read(self, data_file, ds_name, roi):

        c = len(data_file[ds_name].shape) - self.ndims

        array = np.asarray(data_file[ds_name][roi.to_slices() + (slice(None),) * c])
        array = np.transpose(array,
                             axes=[i + self.ndims for i in range(c)] + list(range(self.ndims)))

        return array

    def __repr__(self):

        return self.filename
