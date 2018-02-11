import copy
import logging
import numpy as np

from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.ext import pyklb
from gunpowder.profiling import Timing
from gunpowder.roi import Roi
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from .batch_provider import BatchProvider

logger = logging.getLogger(__name__)

class KlbSource(BatchProvider):
    '''A KLB data source.

    Provides a single array from the given KLB dataset.

    Args:

        filename (string): The KLB file.

        array (ArrayKey): ArrayKey that this source offers.

        array_spec (ArraySpec, optional): An optional :class:`ArraySpec` to
            overwrite the array specs automatically determined from the KLB
            file. This is useful to set ``voxel_size``, for example. Only
            fields that are not ``None`` in the given :class:`ArraySpec` will
            be used.
    '''

    def __init__(
            self,
            filename,
            array,
            array_spec=None):

        self.filename = filename
        self.array = array
        self.array_spec = array_spec

        self.ndims = None

    def setup(self):

        header = pyklb.readheader(self.filename)

        spec = self.__read_spec(header)
        self.provides(self.array, spec)

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

        _, request_spec = request.array_specs.items()[0]

        logger.debug("Reading %s in %s...", self.array, request_spec.roi)

        voxel_size = self.spec[self.array].voxel_size

        # scale request roi to voxel units
        dataset_roi = request_spec.roi/voxel_size

        # shift request roi into dataset
        dataset_roi = dataset_roi - self.spec[self.array].roi.get_offset()/voxel_size

        # create array spec
        array_spec = self.spec[self.array].copy()
        array_spec.roi = request_spec.roi

        # add array to batch
        batch.arrays[self.array] = Array(
            self.__read(dataset_roi),
            array_spec)

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __read_spec(self, header):

        size = header['imagesize_tczyx']

        # strip leading 1 dimensions
        while size[0] == 1 and len(size) > 1:
            size = size[1:]

        dims = Coordinate(size)

        if self.ndims is None:
            self.ndims = len(dims)
        else:
            assert self.ndims == len(dims)

        if self.array_spec is not None:
            spec = self.array_spec
        else:
            spec = ArraySpec()

        if spec.voxel_size is None:
            spec.voxel_size = Coordinate(header['pixelspacing_tczyx'][-self.ndims:])

        if spec.roi is None:
            offset = Coordinate((0,)*self.ndims)
            spec.roi = Roi(offset, dims*spec.voxel_size)

        if spec.dtype is not None:
            assert spec.dtype == header['datatype'], (
                "dtype %s provided in array_specs for %s, but differs from "
                "dataset dtype %s"%(
                    self.array_specs[self.array].dtype, self.array,
                    dataset.dtype))
        else:
            spec.dtype = header['datatype']

        if spec.interpolatable is None:

            spec.interpolatable = spec.dtype in [
                np.float,
                np.float32,
                np.float64,
                np.float128,
                np.uint8 # assuming this is not used for labels
            ]
            logger.warning("WARNING: You didn't set 'interpolatable' for %s. "
                           "Based on the dtype %s, it has been set to %s. "
                           "This might not be what you want.",
                           self.array, spec.dtype, spec.interpolatable)

        return spec

    def __read(self, roi):

        # pyklb reads max-inclusive, gunpowder rois are max exclusive ->
        # subtract (1, 1, ...) from max coordinate
        return pyklb.readroi(
            self.filename,
            roi.get_begin(),
            roi.get_end() - (1,)*self.ndims)

    def __repr__(self):

        return self.filename
