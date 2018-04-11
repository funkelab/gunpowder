import copy
import logging
import numpy as np
import glob

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
    '''A `KLB <https://bitbucket.org/fernandoamat/keller-lab-block-filetype>`_
    data source.

    Provides a single array from the given KLB dataset.

    Args:

        filename (``string``):

            The name of the KLB file. This string can be a glob expression
            (e.g., ``frame_*.klb``), in which case all files that match are
            sorted and stacked together to form an additional dimension (like
            time). The additional dimension will start at 0 and have a default
            voxel size of 1 (which can be overwritten using the ``array_spec``
            argument).

        array (:class:`ArrayKey`):

            ArrayKey that this source offers.

        array_spec (:class:`ArraySpec`, optional):

            An optional :class:`ArraySpec` to overwrite the array specs
            automatically determined from the KLB file. This is useful to set
            ``voxel_size``, for example. Only fields that are not ``None`` in
            the given :class:`ArraySpec` will be used.
    '''

    def __init__(
            self,
            filename,
            array,
            array_spec=None):

        self.filename = filename
        self.array = array
        self.array_spec = array_spec

        self.files = None
        self.ndims = None

    def setup(self):

        self.files = glob.glob(self.filename)
        self.files.sort()

        logger.info("Reading KLB headers of %d files...", len(self.files))
        headers = [ pyklb.readheader(f) for f in self.files ]
        spec = self.__read_spec(headers)

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

    def __read_spec(self, headers):

        num_files = len(headers)
        assert num_files > 0
        common_header = headers[0]
        for header in headers:
            for attr in ['imagesize_tczyx', 'pixelspacing_tczyx']:
                assert (common_header[attr] == header[attr]).all(), (
                    "Headers of provided KLB files differ in attribute %s"%attr)
            assert common_header['datatype'] == header['datatype'], (
                "Headers of provided KLB files differ in attribute datatype")

        size = Coordinate(common_header['imagesize_tczyx'])
        voxel_size = Coordinate(common_header['pixelspacing_tczyx'])
        dtype = common_header['datatype']

        # strip leading 1 dimensions
        while size[0] == 1 and len(size) > 1:
            size = size[1:]
            voxel_size = voxel_size[1:]

        # append num_files dimension
        if num_files > 1:
            size = (num_files,) + size
            voxel_size = (1,) + voxel_size

        dims = Coordinate(size)
        self.ndims = len(dims)

        if self.array_spec is not None:
            spec = self.array_spec
        else:
            spec = ArraySpec()

        if spec.voxel_size is None:
            spec.voxel_size = Coordinate(voxel_size)

        if spec.roi is None:
            offset = Coordinate((0,)*self.ndims)
            spec.roi = Roi(offset, dims*spec.voxel_size)

        if spec.dtype is not None:
            assert spec.dtype == dtype, (
                "dtype %s provided in array_specs for %s, but differs from "
                "dataset dtype %s"%(
                    self.array_specs[self.array].dtype, self.array,
                    dataset.dtype))
        else:
            spec.dtype = dtype

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

        if len(self.files) == 1:

            return self.__read_file(self.files[0], roi)

        else:

            file_indices = range(
                roi.get_begin()[0],
                roi.get_end()[0])

            file_roi = Roi(
                roi.get_begin()[1:],
                roi.get_shape()[1:])

            return np.array([
                    self.__read_file(self.files[i], file_roi)
                    for i in file_indices
                ])

    def __read_file(self, filename, roi):

        # pyklb reads max-inclusive, gunpowder rois are max exclusive ->
        # subtract (1, 1, ...) from max coordinate
        return pyklb.readroi(
            filename,
            roi.get_begin(),
            roi.get_end() - (1,)*roi.dims())

    def __repr__(self):

        return self.filename
