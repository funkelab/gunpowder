import copy
from batch_filter import BatchFilter

class Reject(BatchFilter):

    def __init__(self, max_masked=0.5):
        self.max_masked = max_masked

    def request_batch(self, batch_spec):

        batch_spec.with_gt_mask = True

        have_good_batch = False
        while not have_good_batch:
            batch = self.get_upstream_provider().request_batch(copy.copy(batch_spec))
            have_good_batch = batch.gt_mask.mean()<self.max_masked

        return batch
