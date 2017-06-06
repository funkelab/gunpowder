from random import randint
from skimage.transform import integral_image, integrate
import logging
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.batch_request import BatchRequest
from gunpowder.coordinate import Coordinate
from gunpowder.volume import VolumeType

logger = logging.getLogger(__name__)

class RandomLocation(BatchFilter):
    '''Choses a batch at a random location in the bounding box of the upstream 
    provider.

    The random location is chosen such that the batch request roi lies entirely 
    inside the provider's roi.
    '''

    def __init__(self, min_masked=0, mask_volume_type=VolumeType.GT_MASK):
        '''Create a random location sampler.

        If `min_masked` (and optionally `mask_volume_type`) are set, only 
        batches are returned that have at least the given ratio of masked-in 
        voxels. This is in general faster than using the ``Reject`` node, at the 
        expense of storing an integral volume of the complete mask.

        Args:

            min_masked: If non-zero, require that the random sample contains at 
            least that ratio of masked-in voxels.

            mask_volume_type: The volume type to use for mask checks.
        '''
        self.min_masked = min_masked
        self.mask_volume_type = mask_volume_type


    def setup(self):

        self.roi = self.get_spec().get_total_roi()

        if self.roi is None:
            raise RuntimeError("Can not draw random samples from a provider that does not have a bounding box.")

        if self.min_masked > 0:

            assert self.mask_volume_type in self.get_spec().volumes, "Upstream provider does not have %s"%self.mask_volume_type
            self.mask_roi = self.get_spec().volumes[self.mask_volume_type]

            logger.info("requesting complete mask...")

            mask_request = BatchRequest({self.mask_volume_type: self.mask_roi})
            mask_batch = self.get_upstream_provider().request_batch(mask_request)

            logger.info("allocating mask integral volume...")

            mask_data = mask_batch.volumes[self.mask_volume_type].data
            mask_integral_dtype = np.uint64
            logger.debug("mask size is " + str(mask_data.size))
            if mask_data.size < 2**32:
                mask_integral_dtype = np.uint32
            if mask_data.size < 2**16:
                mask_integral_dtype = np.uint16
            logger.debug("chose %s as integral volume dtype"%mask_integral_dtype)

            self.mask_integral = np.array(mask_data>0, dtype=mask_integral_dtype)
            self.mask_integral = integral_image(self.mask_integral)


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

        good_location_found = False
        while not good_location_found:

            # select a random point inside ROI
            random_offset = Coordinate(
                    randint(begin, end-1)
                    for begin, end in zip(target_roi.get_begin(), target_roi.get_end())
                    )
            logger.debug("random starting point: " + str(random_offset))

            if self.min_masked > 0:
                # get randomly chosen mask ROI
                request_mask_roi = request.volumes[self.mask_volume_type]
                diff = random_offset - request_mask_roi.get_offset()
                request_mask_roi = request_mask_roi.shift(diff)

                # get coordinates inside mask volume
                request_mask_roi_in_volume = request_mask_roi.shift(-self.mask_roi.get_offset())

                # get number of masked-in voxels
                num_masked_in = integrate(
                        self.mask_integral,
                        [request_mask_roi_in_volume.get_begin()],
                        [request_mask_roi_in_volume.get_end()-(1,)*self.mask_integral.ndim]
                )[0]

                mask_ratio = float(num_masked_in)/request_mask_roi.size()
                logger.debug("mask ratio is %f"%mask_ratio)

                if mask_ratio >= self.min_masked:
                    logger.debug("good batch found")
                    good_location_found = True
                else:
                    logger.debug("bad batch found")

            else:
                good_location_found = True

        # shift request ROIs
        diff = random_offset - request_roi.get_offset()
        for (volume_type, roi) in request.volumes.items():
            roi = roi.shift(diff)
            logger.debug("new %s ROI: %s"%(volume_type, roi))
            request.volumes[volume_type] = roi
            assert self.roi.contains(roi)

        for (points_type, roi) in request.points.items():
            roi = roi.shift(diff)
            logger.debug("new %s ROI: %s"%(points_type, roi))
            request.points[points_type] = roi
            assert self.roi.contains(roi)


    def process(self, batch, request):

        # reset ROIs to request
        for (volume_type,roi) in request.volumes.items():
            batch.volumes[volume_type].roi = roi
        for (points_type, roi) in request.points.items():
            batch.points[points_type].roi = roi

