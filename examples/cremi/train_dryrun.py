from __future__ import print_function

import math
import numpy as np

from gunpowder.ext import malis
from gunpowder import *

class DummyTrain(BatchFilter):
    def process(self, batch):
        batch.prediction = np.zeros(batch.spec.output_roi.get_shape())

def train():

    set_verbose()

    affinity_neighborhood = malis.mknhood3d()

    data_source_trees = tuple(
        Hdf5Source(
            sample,
            raw_dataset='volumes/raw',
            gt_dataset='volumes/labels/neuron_ids'
        ) +
        Normalize() +
        RandomLocation()
        for sample in ['sample_A_20160501.hdf', 'sample_B_20160501.hdf', 'sample_C_20160501.hdf']
    )

    batch_provider_tree = (
        data_source_trees +
        RandomProvider() +
        ExcludeLabels([416759, 397008], 8) +
        ElasticAugmentation([4,40,40], [0,2,2], [0,math.pi/2.0]) +
        SimpleAugment(transpose_only_xy=True) +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        DefectAugment(prob_missing=0.1, prob_low_contrast=0.1, contrast_scale=0.1) +
        GrowBoundary(steps=3, only_xy=True) +
        AddGtAffinities(affinity_neighborhood) +
        ZeroOutConstSections() +
        PreCache(
            lambda : batch_spec,
            cache_size=10,
            num_workers=5) +
        DummyTrain() +
        Snapshot(every=1)
    )

    n = 10
    print("Training for", n, "iterations")

    batch_spec = BatchSpec(
        (84,268,268),
        (56,56,56),
        with_gt=True,
        with_gt_mask=False,
        with_gt_affinities=True
    )

    with build(batch_provider_tree) as minibatch_maker:
        for i in range(n):
            minibatch_maker.request_batch(batch_spec)

    print("Finished")


if __name__ == "__main__":
    train()
