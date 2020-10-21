from .batch_filter import BatchFilter
from gunpowder.array import ArrayKey, Array
from gunpowder.array_spec import ArraySpec
from gunpowder.batch_request import BatchRequest
from gunpowder.batch import Batch
import logging
import numbers
import numpy as np

logger = logging.getLogger(__name__)

class UpSample(BatchFilter):
    '''Upsample arrays in a batch by given factors.

    Args:

        source (:class:`ArrayKey`): 
            The key of the array to upsample.

        factor (``int`` or ``tuple`` of ``int``):

            The factor to upsample with.

        target (:class:`ArrayKey`):

            The key of the array to store the upsampled ``source``.
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

        if not isinstance(self.factor, tuple):
            self.factor = (self.factor,)*spec.roi.dims()

        assert spec.voxel_size % self.factor == (0,)*len(spec.voxel_size), \
            "voxel size of upsampled volume is not integer: %s/%s = %s" % (
                spec.voxel_size,
                self.factor,
                tuple(v/f for v, f in zip(spec.voxel_size, self.factor)))
        spec.voxel_size /= self.factor
        self.provides(self.target, spec)

    def prepare(self, request):
        deps = BatchRequest()

        if self.target not in request:
            return

        logger.debug("preparing upsampling of " + str(self.source))

        request_roi = request[self.target].roi
        logger.debug("request ROI is %s"%request_roi)
        # expand to source voxel size
        source_voxel_size = self.spec[self.source].voxel_size
        request_roi = request_roi.snap_to_grid(source_voxel_size, mode='grow')
        logger.debug("expanded request ROI is %s"%request_roi)

        # add or merge to batch request
        deps[self.source] = ArraySpec(roi=request_roi)

        return deps


    def process(self, batch, request):
        outputs = Batch()

        if self.target not in request:
            return

        input_roi = batch.arrays[self.source].spec.roi
        request_roi = request[self.target].roi

        # get roi expanded to source voxel size
        source_voxel_size = self.spec[self.source].voxel_size
        expanded_request_roi = request_roi.snap_to_grid(source_voxel_size, mode='grow')
        logger.debug("expanded request ROI is %s"%request_roi)
        assert input_roi.contains(expanded_request_roi)

        # crop to necessary region
        crop = batch.arrays[self.source].crop(expanded_request_roi)
        data = crop.data

        # upsample
        logger.debug("upsampling %s with %s", self.source, self.factor)
        for d, f in enumerate(self.factor):
            data = np.repeat(data, f, axis=d)

        # create output array
        spec = self.spec[self.target].copy()
        spec.roi = expanded_request_roi
        expanded_array = Array(data, spec)
        array = expanded_array.crop(request_roi)
        outputs.arrays[self.target] = array
        return outputs
