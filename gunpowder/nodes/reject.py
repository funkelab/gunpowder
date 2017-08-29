import logging

from .batch_filter import BatchFilter
from gunpowder.profiling import Timing
from gunpowder.volume import VolumeTypes

logger = logging.getLogger(__name__)

class Reject(BatchFilter):

    def __init__(self, min_masked=0.5, mask_volume_type=VolumeTypes.GT_MASK):
        self.min_masked = min_masked
        self.mask_volume_type = mask_volume_type

    def setup(self):
        assert self.mask_volume_type in self.spec, "Reject can only be used if %s is provided"%self.mask_volume_type
        self.upstream_provider = self.get_upstream_provider()

    def provide(self, request):

        report_next_timeout = 10
        num_rejected = 0

        timing = Timing(self)
        timing.start()

        assert self.mask_volume_type in request, "Reject can only be used if a GT mask is requested"

        have_good_batch = False
        while not have_good_batch:

            batch = self.upstream_provider.request_batch(request)
            mask_ratio = batch.volumes[self.mask_volume_type].data.mean()
            have_good_batch = mask_ratio>=self.min_masked

            if not have_good_batch:

                logger.debug(
                    "reject batch with mask ratio %f at "%mask_ratio +
                    str(batch.volumes[self.mask_volume_type].spec.roi))
                num_rejected += 1

                if timing.elapsed() > report_next_timeout:

                    logger.warning("rejected %d batches, been waiting for a good one since %ds"%(num_rejected, report_next_timeout))
                    report_next_timeout *= 2

        logger.debug(
            "good batch with mask ratio %f found at "%mask_ratio +
            str(batch.volumes[self.mask_volume_type].spec.roi))

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
