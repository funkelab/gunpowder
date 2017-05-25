from __future__ import print_function

import math
import numpy as np
import random

from gunpowder import *

class DummyTrain(BatchFilter):
    def process(self, batch):
        pass

def train():

    random.seed(42)
    set_verbose()

    affinity_neighborhood = np.array([[-1,0,0],[0,-1,0],[0,0,-1]])

    request = BatchRequest()
    request.add_volume_request(VolumeType.RAW, (84,268,268))
    request.add_volume_request(VolumeType.GT_LABELS, (56,56,56))
    request.add_volume_request(VolumeType.GT_MASK, (56,56,56))
    request.add_volume_request(VolumeType.GT_IGNORE, (56,56,56))
    request.add_volume_request(VolumeType.GT_AFFINITIES, (56,56,56))

    data_sources = tuple(
        Hdf5Source(
            sample,
            raw_dataset='volumes/raw',
            gt_dataset='volumes/labels/neuron_ids',
            gt_mask_dataset='volumes/labels/mask',
        ) +
        Padding(
            {
                VolumeType.RAW: (100,100,100),
                VolumeType.GT_LABELS: (100,100,100),
                VolumeType.GT_MASK: (100,100,100)
            },
            {
                VolumeType.RAW: 255,
                VolumeType.GT_LABELS: 23
            }
        ) +
        Normalize() +
        RandomLocation() +
        Reject(0.9)
        for sample in ['sample_A.hdf']
    )

    batch_provider_tree = (
        data_sources +
        Snapshot(every=1, output_filename='00.hdf') +
        RandomProvider() +
        ExcludeLabels([8094], ignore_mask_erode=12) +
        Snapshot(every=1, output_filename='01.hdf') +
        # ElasticAugmentation([4,40,40], [0,2,2], [0,math.pi/2.0]) +
        Snapshot(every=1, output_filename='02.hdf') +
        SimpleAugment(transpose_only_xy=True) +
        Snapshot(every=1, output_filename='03.hdf') +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        DefectAugment(prob_missing=0.1, prob_low_contrast=0.1, contrast_scale=0.1) +
        Snapshot(every=1, output_filename='04.hdf') +
        GrowBoundary(steps=3, only_xy=True) +
        Snapshot(every=1, output_filename='05.hdf') +
        AddGtAffinities(affinity_neighborhood) +
        ZeroOutConstSections() +
        # PreCache(
            # request,
            # cache_size=10,
            # num_workers=5) +
        DummyTrain() +
        Snapshot(every=1, output_filename='06.hdf')
    )

    n = 10
    print("Training for", n, "iterations")

    with build(batch_provider_tree) as minibatch_maker:
        for i in range(n):
            minibatch_maker.request_batch(request)

    print("Finished")


if __name__ == "__main__":
    train()
