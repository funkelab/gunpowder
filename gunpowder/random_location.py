from batch_filter import BatchFilter
from random import randint

import logging
logger = logging.getLogger(__name__)

class RandomLocation(BatchFilter):
    '''Choses a batch at a random location in the bounding box of the upstream 
    provider.

    The random location is chosen such that the batch specs input roi lies 
    /inside/ the provder's roi.
    '''

    def setup(self):

        provider_spec = self.get_upstream_provider().get_spec()
        if provider_spec.roi.get_bounding_box() is None:
            raise RuntimeError("Can not draw random samples from a provider that does not have a bounding box.")

        self.roi = provider_spec.roi

    def prepare(self, batch_spec):

        logger.debug("original input ROI: %s"%batch_spec.input_roi)

        shape = batch_spec.input_roi.get_shape()

        for d in range(self.roi.dims()):
            assert self.roi.get_shape()[d] >= shape[d], "Requested shape %s does not fit into provided ROI %s."%(shape,self.roi)

        target_bb = self.roi.get_bounding_box()

        current_offset = batch_spec.input_roi.get_offset()
        new_offset = tuple(
                randint(target_bb[d].start, target_bb[d].stop - shape[d])
                for d in range(len(shape))
        )
        diff = tuple(
                new_offset[d] - current_offset[d]
                for d in range(batch_spec.input_roi.dims())
        )

        batch_spec.input_roi = batch_spec.input_roi.shift(diff)
        batch_spec.output_roi = batch_spec.output_roi.shift(diff)

        logger.debug("target ROI: %s"%self.roi)
        logger.debug("current offset: %s"%str(current_offset))
        logger.debug("new random offset: %s"%str(new_offset))
        logger.debug("new input ROI: %s"%batch_spec.input_roi)
        logger.debug("new output ROI: %s"%batch_spec.output_roi)

    def process(self, batch):
        pass
