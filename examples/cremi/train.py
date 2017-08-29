from __future__ import print_function

import math
import numpy as np
import random

from gunpowder import *
from gunpowder.caffe import *
from gunpowder.ext import malis

def train():

    affinity_neighborhood = malis.mknhood3d()
    solver_parameters = SolverParameters()
    solver_parameters.train_net = 'net.prototxt'
    solver_parameters.base_lr = 1e-4
    solver_parameters.momentum = 0.95
    solver_parameters.momentum2 = 0.999
    solver_parameters.delta = 1e-8
    solver_parameters.weight_decay = 0.000005
    solver_parameters.lr_policy = 'inv'
    solver_parameters.gamma = 0.0001
    solver_parameters.power = 0.75
    solver_parameters.snapshot = 2000
    solver_parameters.snapshot_prefix = 'net'
    solver_parameters.type = 'Adam'
    solver_parameters.resume_from = None
    solver_parameters.train_state.add_stage('euclid')

    request = BatchRequest()
    request.add_volume_request(VolumeTypes.RAW, (84,268,268))
    request.add_volume_request(VolumeTypes.GT_LABELS, (56,56,56))
    request.add_volume_request(VolumeTypes.GT_MASK, (56,56,56))
    request.add_volume_request(VolumeTypes.GT_IGNORE, (56,56,56))
    request.add_volume_request(VolumeTypes.GT_AFFINITIES, (56,56,56))

    data_sources = tuple(
        Hdf5Source(
            'sample_'+s+'_padded_20160501.aligned.filled.cropped.hdf',
            datasets = {
                VolumeTypes.RAW: 'volumes/raw',
                VolumeTypes.GT_LABELS: 'volumes/labels/neuron_ids_notransparency',
                VolumeTypes.GT_MASK: 'volumes/labels/mask',
            }
        ) +
        Normalize() +
        RandomLocation()
        for s in ['A', 'B', 'C']
    )

    artifact_source = (
        Hdf5Source(
            'sample_ABC_padded_20160501.defects.hdf',
            datasets = {
                VolumeTypes.RAW: 'defect_sections/raw',
                VolumeTypes.ALPHA_MASK: 'defect_sections/mask',
            }
        ) +
        RandomLocation(min_masked=0.05, mask_volume_type=VolumeTypes.ALPHA_MASK) +
        Normalize() +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        ElasticAugment([4,40,40], [0,2,2], [0,math.pi/2.0]) +
        SimpleAugment(transpose_only_xy=True)
    )

    snapshot_request = BatchRequest()
    snapshot_request.add_volume_request(VolumeTypes.LOSS_GRADIENT, (56,56,56))

    batch_provider_tree = (
        data_sources +
        RandomProvider() +
        ExcludeLabels([8094], ignore_mask_erode=12) +
        ElasticAugment([4,40,40], [0,2,2], [0,math.pi/2.0], prob_slip=0.05,prob_shift=0.05,max_misalign=25) +
        SimpleAugment(transpose_only_xy=True) +
        GrowBoundary(steps=3, only_xy=True) +
        AddGtAffinities(affinity_neighborhood) +
        SplitAndRenumberSegmentationLabels() +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        DefectAugment(
            prob_missing=0.03,
            prob_low_contrast=0.01,
            prob_artifact=0.03,
            artifact_source=artifact_source,
            contrast_scale=0.1) +
        ZeroOutConstSections() +
        IntensityScaleShift(2,-1) +
        BalanceAffinityLabels() +
        PreCache(
            cache_size=10,
            num_workers=5) +
        Train(solver_parameters, use_gpu=0) +
        Snapshot(every=10, output_filename='batch_{id}.hdf', additional_request=snapshot_request)
    )

    n = 10
    print("Training for", n, "iterations")

    with build(batch_provider_tree) as minibatch_maker:
        for i in range(n):
            minibatch_maker.request_batch(request)

    print("Finished")

if __name__ == "__main__":
    train()
