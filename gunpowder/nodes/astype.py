from .batch_filter import BatchFilter
from gunpowder.array import ArrayKey, Array
from gunpowder.batch import Batch
import logging

logger = logging.getLogger(__name__)


class AsType(BatchFilter):
    """Cast arrays to a different datatype (ex: np.float32 --> np.uint8).

    Args:

        source (:class:`ArrayKey`):

            The key of the array to cast.

        target_dtype (str or dtype):

            The voxel size of the target.

        target (:class:`ArrayKey`, optional):

            The key of the array to store the cast ``source``.

    """

    def __init__(self, source, target_dtype, target=None):
        assert isinstance(source, ArrayKey)
        if target is not None:
            assert isinstance(target, ArrayKey)
            self.target = target
        else:
            self.target = source

        self.source = source
        self.target_dtype = target_dtype

    def setup(self):
        spec = self.spec[self.source].copy()
        spec.dtype = self.target_dtype
        if self.target is not self.source:
            self.provides(self.target, spec)
        else:
            self.updates(self.source, spec)
        self.enable_autoskip()

    def process(self, batch, request):
        source = batch.arrays[self.source]
        source_data = source.data

        cast_data = source_data.astype(self.target_dtype)

        target_spec = source.spec.copy()
        target_spec.dtype = cast_data.dtype
        target_array = Array(cast_data, target_spec)

        # create output array
        outputs = Batch()
        outputs.arrays[self.target] = target_array

        return outputs
