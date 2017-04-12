import malis
import numpy as np
from batch_filter import BatchFilter

class AddGtAffinities(BatchFilter):

    def __init__(self, affinity_neighborhood):
        self.affinity_neighborhood = affinity_neighborhood

    def process(self, batch):

        # do nothing if no gt affinities were requested
        if not batch.spec.with_gt_affinities:
            return

        # do nothing if gt affinities are already present
        if batch.gt_affinities is not None:
            print("AddGtAffinities: batch already contains affinities, skipping")
            return

        print("AddGtAffinities: computing ground-truth affinities from labels")
        batch.gt_affinities = malis.seg_to_affgraph(
                batch.gt.astype(np.int32),
                self.affinity_neighborhood).astype(np.float32)

        batch.affinity_neighborhood = self.affinity_neighborhood
