import logging
import numpy as np

from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class Normalize(BatchFilter):
    '''Normalize the values of a volume to be floats between 0 and 1, based on
    the type of the volume.
    '''

    def __init__(self, volume, factor=None, dtype=np.float32):

        self.volume = volume
        self.factor = factor
        self.dtype = dtype

    def process(self, batch, request):

        factor = self.factor
        volume = batch.volumes[self.volume]

        if factor is None:

            logger.debug("automatically normalizing %s with dtype=%s",
                    self.volume, volume.data.dtype)

            if volume.data.dtype == np.uint8:
                factor = 1.0/255
            elif volume.data.dtype == np.float32:
                assert volume.data.min() >= 0 and volume.data.max() <= 1, (
                        "Values are float but not in [0,1], I don't know how "
                        "to normalize. Please provide a factor.")
                factor = 1.0
            else:
                raise RuntimeError("Automatic normalization for " +
                        str(volume.data.dtype) + " not implemented, please "
                        "provide a factor.")

        logger.debug("scaling %s with %f", self.volume, factor)
        volume.data = volume.data.astype(self.dtype)*factor
