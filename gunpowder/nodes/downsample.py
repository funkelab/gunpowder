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

        source (:class:`ArrayKey`):

            The key of the array to downsample.

        factor (``int`` or ``tuple`` of ``int``):

            The factor to downsample with.

        target (:class:`ArrayKey`):

            The key of the array to store the downsampled ``source``.
    '''

    def __init__(self, source, factor, target):

        assert isinstance(source, ArrayKey)
        assert isinstance(target, ArrayKey)
        assert (
            isinstance(factor, numbers.Number) or isinstance(factor, tuple)), (
            "Scaling factor should be a number or a tuple of numbers.")

        self.source = source
        self.factor = factor
        self.target = target

    def setup(self):

        spec = self.spec[self.source].copy()
        spec.voxel_size *= self.factor
        self.provides(self.target, spec)

    def prepare(self, request):

        if self.target not in request:
            return

        logger.debug("preparing downsampling of " + str(self.source))

        request_roi = request[self.target].roi
        logger.debug("request ROI is %s"%request_roi)

        # add or merge to batch request
        if self.source in request:
            request[self.source].roi = request[self.source].roi.union(request_roi)
            logger.debug(
                "merging with existing request to %s",
                request[self.source].roi)
        else:
            request[self.source].roi = request_roi
            logger.debug("adding as new request")

    def process(self, batch, request):

        if self.target not in request:
            return

        input_roi = batch.arrays[self.source].spec.roi
        request_roi = request[self.target].roi

        assert input_roi.contains(request_roi)

        # downsample
        if isinstance(self.factor, tuple):
            slices = tuple(
                slice(None, None, k)
                for k in self.factor)
        else:
            slices = tuple(
                slice(None, None, self.factor)
                for i in range(input_roi.dims()))

        logger.debug("downsampling %s with %s", self.source, slices)

        crop = batch.arrays[self.source].crop(request_roi)
        data = crop.data[slices]

        # create output array
        spec = self.spec[self.target].copy()
        spec.roi = request_roi
        batch.arrays[self.target] = Array(data, spec)

        # restore requested rois
        request_roi = request[self.source].roi

        if input_roi != request_roi:

            assert input_roi.contains(request_roi)

            logger.debug(
                "restoring original request roi %s of %s from %s",
                request_roi, self.source, input_roi)
            cropped = batch.arrays[self.source].crop(request_roi)
            batch.arrays[self.source] = cropped
