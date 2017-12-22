import logging

from .batch_filter import BatchFilter
from gunpowder.profiling import Timing

logger = logging.getLogger(__name__)

class Reject(BatchFilter):
    '''Reject batches based on the masked-in vs. masked-out ratio.

    Args:

        mask(:class:``ArrayType``): The mask to use.

        min_masked(float, optional): The minimal required ratio of masked-in
            vs. masked-out voxels. Defaults to 0.5.
    '''

    def __init__(self, mask, min_masked=0.5):
        self.mask = mask
        self.min_masked = min_masked

    def setup(self):
        assert self.mask in self.spec, "Reject can only be used if %s is provided"%self.mask
        self.upstream_provider = self.get_upstream_provider()

    def provide(self, request):

        report_next_timeout = 10
        num_rejected = 0

        timing = Timing(self)
        timing.start()

        assert self.mask in request, "Reject can only be used if a GT mask is requested"

        have_good_batch = False
        while not have_good_batch:

            batch = self.upstream_provider.request_batch(request)
            mask_ratio = batch.arrays[self.mask].data.mean()
            have_good_batch = mask_ratio>=self.min_masked

            if not have_good_batch:

                logger.debug(
                    "reject batch with mask ratio %f at "%mask_ratio +
                    str(batch.arrays[self.mask].spec.roi))
                num_rejected += 1

                if timing.elapsed() > report_next_timeout:

                    logger.warning("rejected %d batches, been waiting for a good one since %ds"%(num_rejected, report_next_timeout))
                    report_next_timeout *= 2

        logger.debug(
            "good batch with mask ratio %f found at "%mask_ratio +
            str(batch.arrays[self.mask].spec.roi))

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
