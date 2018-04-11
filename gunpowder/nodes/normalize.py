import logging
import numpy as np

from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class Normalize(BatchFilter):
    '''Normalize the values of an array to be floats between 0 and 1, based on
    the type of the array.

    Args:

        array (:class:`ArrayKey`):

            The key of the array to modify.

        factor (scalar, optional):

            The factor to use. If not given, a factor is chosen based on the
            ``dtype`` of the array (e.g., ``np.uint8`` would result in a factor
            of ``1.0/255``).

        dtype (data-type, optional):

            The datatype of the normalized array. Defaults to ``np.float32``.
    '''

    def __init__(self, array, factor=None, dtype=np.float32):

        self.array = array
        self.factor = factor
        self.dtype = dtype

    def process(self, batch, request):

        if self.array not in batch.arrays:
            return

        factor = self.factor
        array = batch.arrays[self.array]

        if factor is None:

            logger.debug("automatically normalizing %s with dtype=%s",
                    self.array, array.data.dtype)

            if array.data.dtype == np.uint8:
                factor = 1.0/255
            elif array.data.dtype == np.uint16:
                factor = 1.0/(255*255)
            elif array.data.dtype == np.float32:
                assert array.data.min() >= 0 and array.data.max() <= 1, (
                        "Values are float but not in [0,1], I don't know how "
                        "to normalize. Please provide a factor.")
                factor = 1.0
            else:
                raise RuntimeError("Automatic normalization for " +
                        str(array.data.dtype) + " not implemented, please "
                        "provide a factor.")

        logger.debug("scaling %s with %f", self.array, factor)
        array.data = array.data.astype(self.dtype)*factor
