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

        shift_roi = None

        for volume_type, request_roi in request.volumes.items():

            assert volume_type in self.get_spec().volumes, "Requested %s, but source does not provide it."%volume_type
            provided_roi = self.get_spec().volumes[volume_type]

            volume_shift_roi = provided_roi.shift(-request_roi.get_begin()).grow((0,0,0), -request_roi.get_shape())

            if shift_roi is None:
                shift_roi = volume_shift_roi
            else:
                shift_roi = shift_roi.intersect(volume_shift_roi)

        logger.debug("valid shifts for request in " + str(shift_roi))

        assert shift_roi.size() > 0, "Can not satisfy batch request, no location covers all requested ROIs."

        good_location_found = False
        while not good_location_found:

            # select a random point inside ROI
            random_shift = Coordinate(
                    randint(begin, end-1)
                    for begin, end in zip(shift_roi.get_begin(), shift_roi.get_end())
            )

            logger.debug("random shift: " + str(random_shift))

            if self.min_masked > 0:

                # get randomly chosen mask ROI
                request_mask_roi = request.volumes[self.mask_volume_type]
                request_mask_roi = request_mask_roi.shift(random_shift)

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
        for (volume_type, roi) in request.volumes.items():
            roi = roi.shift(random_shift)
            logger.debug("new %s ROI: %s"%(volume_type,roi))
            request.volumes[volume_type] = roi
            assert self.roi.contains(roi)

    def process(self, batch, request):

        # reset ROIs to request
        for (volume_type,roi) in request.volumes.items():
            batch.volumes[volume_type].roi = roi
