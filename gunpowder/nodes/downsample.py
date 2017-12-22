from .batch_filter import BatchFilter
from gunpowder.coordinate import Coordinate
from gunpowder.array import ArrayKey, Array
import logging
import numbers
import numpy as np

logger = logging.getLogger(__name__)

class DownSample(BatchFilter):
    '''Downsample arrays in a batch by given factors.

    Args:

        array_factors (dict): Dictionary mapping target :class:`ArrayKey` to 
            a tuple `(f, array_type)` of downsampling factor `f` and source 
            :class:`ArrayKey`. `f` can be a single integer or a tuple of 
            integers, one for each dimension of the array to downsample.
    '''

    def __init__(self, array_factors):

        self.array_factors = array_factors

        for output_array, downsample in array_factors.items():

            assert isinstance(output_array, ArrayKey)
            assert isinstance(downsample, tuple)
            assert len(downsample) == 2
            f, input_array = downsample
            assert isinstance(input_array, ArrayKey)
            assert isinstance(f, numbers.Number) or isinstance(f, tuple), "Scaling factor should be a number or a tuple of numbers."

    def setup(self):

        for output_array, downsample in self.array_factors.items():
            f, input_array = downsample
            spec = self.spec[input_array].copy()
            spec.voxel_size *= f
            self.provides(output_array, spec)

    def prepare(self, request):

        for output_array, downsample in self.array_factors.items():

            f, input_array = downsample

            if output_array not in request:
                continue

            logger.debug("preparing downsampling of " + str(input_array))

            request_roi = request[output_array].roi
            logger.debug("request ROI is %s"%request_roi)

            # add or merge to batch request
            if input_array in request:
                request[input_array].roi = request[input_array].roi.union(request_roi)
                logger.debug("merging with existing request to %s"%request[input_array].roi)
            else:
                request[input_array].roi = request_roi
                logger.debug("adding as new request")

    def process(self, batch, request):

        for output_array, downsample in self.array_factors.items():

            f, input_array = downsample

            if output_array not in request:
                continue

            input_roi = batch.arrays[input_array].spec.roi
            request_roi = request[output_array].roi

            assert input_roi.contains(request_roi)

            # downsample
            if isinstance(f, tuple):
                slices = tuple(slice(None, None, k) for k in f)
            else:
                slices = tuple(slice(None, None, f) for i in range(input_roi.dims()))

            logger.debug("downsampling %s with %s"%(input_array, slices))

            crop = batch.arrays[input_array].crop(request_roi)
            data = crop.data[slices]

            # create output array
            spec = self.spec[output_array].copy()
            spec.roi = request_roi
            batch.arrays[output_array] = Array(data, spec)

        # restore requested rois
        for output_array, downsample in self.array_factors.items():

            f, input_array = downsample
            if input_array not in batch.arrays:
                continue

            input_roi = batch.arrays[input_array].spec.roi
            request_roi = request[input_array].roi

            if input_roi != request_roi:

                assert input_roi.contains(request_roi)

                logger.debug("restoring original request roi %s of %s from %s"%(request_roi, input_array, input_roi))
                batch.arrays[input_array] = batch.arrays[input_array].crop(request_roi)
