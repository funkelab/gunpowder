import logging
from random import randint

from .batch_filter import BatchFilter
from gunpowder.coordinate import Coordinate

logger = logging.getLogger(__name__)

class RandomLocation(BatchFilter):
    '''Choses a batch at a random location in the bounding box of the upstream 
    provider.

    The random location is chosen such that the batch request roi lies entirely 
    inside the provder's roi.
    '''

    def setup(self):

        self.roi = self.get_spec().get_total_roi()
        if self.roi is None:
            raise RuntimeError("Can not draw random samples from a provider that does not have a bounding box.")

    def prepare(self, request):

        request_roi = request.get_total_roi()
        logger.debug("total requested ROI: %s"%request_roi)

        shape = request_roi.get_shape()
        for d in range(self.roi.dims()):
            assert self.roi.get_shape()[d] >= shape[d], "Requested shape %s does not fit into provided ROI %s."%(shape,self.roi)

        target_roi = self.roi
        logger.debug("valid target ROI to fit total request ROI: " + str(target_roi))

        # shrink target ROI, such that it contains only valid offset positions 
        # for request ROI
        target_roi = target_roi.grow(None, -request_roi.get_shape())

        logger.debug("valid starting points for request in " + str(target_roi))

        # select a random point inside ROI
        random_offset = Coordinate(
                randint(begin, end-1)
                for begin, end in zip(target_roi.get_begin(), target_roi.get_end())
        )

        logger.debug("random starting point: " + str(random_offset))

        # shift request ROIs
        diff = random_offset - request_roi.get_offset()
        for (volume_type, roi) in request.volumes.items():
            roi = roi.shift(diff)
            logger.debug("new %s ROI: %s"%(volume_type,roi))
            request.volumes[volume_type] = roi
            assert self.roi.contains(roi)

    def process(self, batch, request):

        # reset ROIs to request
        for (volume_type,roi) in request.volumes.items():
            batch.volumes[volume_type].roi = roi
