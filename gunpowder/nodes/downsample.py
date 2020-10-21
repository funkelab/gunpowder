from .batch_filter import BatchFilter
from gunpowder.array import ArrayKey, Array
from gunpowder.batch_request import BatchRequest
from gunpowder.batch import Batch
import logging
import numbers

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
        self.enable_autoskip()

    def prepare(self, request):

        deps = BatchRequest()
        deps[self.source] = request[self.target]
        return deps

    def process(self, batch, request):
        outputs = Batch()

        # downsample
        if isinstance(self.factor, tuple):
            slices = tuple(
                slice(None, None, k)
                for k in self.factor)
        else:
            slices = tuple(
                slice(None, None, self.factor)
                for i in range(batch[self.source].spec.roi.dims()))

        logger.debug("downsampling %s with %s", self.source, slices)

        data = batch.arrays[self.source].data[slices]

        # create output array
        spec = self.spec[self.target].copy()
        spec.roi = request[self.target].roi
        outputs.arrays[self.target] = Array(data, spec)
        
        return outputs
