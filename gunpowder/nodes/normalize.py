import logging
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.volume import VolumeType

logger = logging.getLogger(__name__)

class Normalize(BatchFilter):
    '''Normalize the raw volume to values between 0 and 1.
    '''

    def __init__(self, factor=None, dtype=np.float32):

        self.factor = factor
        self.dtype = dtype

    def process(self, batch, request):

        factor = self.factor
        raw = batch.volumes[VolumeType.RAW]

        if factor is None:

            logger.debug("automatically normalizing raw data with dtype=" + str(raw.data.dtype))

            if raw.data.dtype == np.uint8:
                factor = 1.0/255
            elif raw.data.dtype == np.float32:
                assert raw.data.min() >= 0 and raw.data.max() <= 1, "Raw values are float but not in [0,1], I don't know how to normalize. Please provide a factor."
                factor = 1.0
            else:
                raise RuntimeError("Automatic normalization for " + str(raw.data.dtype) + " not implemented, please provide a factor.")

        logger.debug("scaling raw data with " + str(factor))
        raw.data = raw.data.astype(self.dtype)*factor
