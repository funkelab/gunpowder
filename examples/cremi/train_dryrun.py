from __future__ import print_function

import math
import numpy as np
import random

from gunpowder import *

class DummyTrain(BatchFilter):
    def process(self, batch):
        print("DummyTrain: input ROI of batch going downstream: " + str(batch.spec.input_roi))
        batch.prediction = np.zeros(batch.spec.output_roi.get_shape())

def train():

    random.seed(42)
    set_verbose()

    affinity_neighborhood = np.array([[-1,0,0],[0,-1,0],[0,0,-1]])

    data_source_trees = tuple(
        Hdf5Source(
            sample,
            raw_dataset='volumes/raw',
            gt_dataset='volumes/labels/neuron_ids',
            gt_mask_dataset='volumes/labels/mask',
        ) +
        Snapshot(every=1, output_filename='00.hdf') +
        Normalize() +
        RandomLocation() +
        Reject()
        for sample in ['sample_A.hdf']
    )

    batch_provider_tree = (
        data_source_trees +
        RandomProvider() +
        ExcludeLabels([416759, 397008], 8) +
        Snapshot(every=1, output_filename='01.hdf') +
        ElasticAugmentation([4,40,40], [0,2,2], [0,math.pi/2.0]) +
        Snapshot(every=1, output_filename='02.hdf') +
        SimpleAugment(transpose_only_xy=True) +
        Snapshot(every=1, output_filename='03.hdf') +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        DefectAugment(prob_missing=0.1, prob_low_contrast=0.1, contrast_scale=0.1) +
        Snapshot(every=1, output_filename='04.hdf') +
        GrowBoundary(steps=3, only_xy=True) +
        Snapshot(every=1, output_filename='05.hdf') +
        # AddGtAffinities(affinity_neighborhood) +
        ZeroOutConstSections() +
        # PreCache(
            # lambda : batch_spec,
            # cache_size=10,
            # num_workers=5) +
        DummyTrain() +
        Snapshot(every=1, output_filename='06.hdf')
    )

    n = 1
    print("Training for", n, "iterations")

    batch_spec = BatchSpec(
        (84,268,268),
        (56,56,56),
        with_volumes = [VolumeType.RAW, VolumeType.GT_LABELS, VolumeType.GT_AFFINITIES]
    )

    with build(batch_provider_tree) as minibatch_maker:
        for i in range(n):
            minibatch_maker.request_batch(batch_spec)

    print("Finished")


if __name__ == "__main__":
    train()
