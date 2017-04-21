import copy
from batch_filter import BatchFilter
from profiling import Timing

import logging
logger = logging.getLogger(__name__)

class Reject(BatchFilter):

    def __init__(self, min_masked=0.5):
        self.min_masked = min_masked

    def setup(self):
        assert self.get_upstream_provider().get_spec().has_gt_mask, "Reject can only be used if GT masks are provided"

    def request_batch(self, batch_spec):

        timing = Timing(self)
        timing.start()

        batch_spec.with_gt_mask = True

        have_good_batch = False
        while not have_good_batch:
            batch = self.get_upstream_provider().request_batch(copy.copy(batch_spec))
            mask_ratio = batch.gt_mask.mean()
            have_good_batch = mask_ratio>=self.min_masked
            if not have_good_batch:
                logger.debug("reject batch with mask ratio %f at "%mask_ratio + str(batch.spec.output_roi.get_bounding_box()))

        logger.debug("good batch with mask ratio %f found at "%mask_ratio + str(batch.spec.output_roi.get_bounding_box()))

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
