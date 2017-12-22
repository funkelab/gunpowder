from __future__ import print_function

import math
import numpy as np
import random

from gunpowder import *

def train():

    random.seed(42)
    set_verbose()

    affinity_neighborhood = np.array([[-1,0,0],[0,-1,0],[0,0,-1]])
    register_array_type('GT_LABELS_2')
    register_array_type('GT_LABELS_4')
    register_array_type('GT_BOUNDARY_GRADIENT')
    register_array_type('GT_BOUNDARY_DISTANCE')
    register_array_type('GT_BOUNDARY')
    n = 35

    request = BatchRequest()
    request.add(ArrayKeys.RAW, Coordinate((84,268,268))*(40,4,4))
    request.add(ArrayKeys.GT_LABELS, Coordinate((56,56,56))*(40,4,4))
    request.add(ArrayKeys.GT_LABELS_2, Coordinate((56,56,56))*(40,4,4))
    request.add(ArrayKeys.GT_LABELS_4, Coordinate((56,56,56))*(40,4,4))
    request.add(ArrayKeys.GT_IGNORE, Coordinate((56,56,56))*(40,4,4))
    request.add(ArrayKeys.GT_AFFINITIES, Coordinate((56,56,56))*(40,4,4))
    request.add(ArrayKeys.GT_BOUNDARY_GRADIENT, Coordinate((56,56,56))*(40,4,4))
    request.add(ArrayKeys.GT_BOUNDARY_DISTANCE, Coordinate((56,56,56))*(40,4,4))
    request.add(ArrayKeys.GT_BOUNDARY, Coordinate((56,56,56))*(40,4,4))
    request.add(ArrayKeys.LOSS_SCALE, Coordinate((56,56,56))*(40,4,4))

    data_sources = tuple(
        Hdf5Source(
            sample,
            datasets = {
                ArrayKeys.RAW: 'volumes/raw',
                ArrayKeys.GT_LABELS: 'volumes/labels/neuron_ids',
            }
        ) +
        Normalize() +
        RandomLocation()
        for sample in ['sample_A_20160501.hdf','sample_B_20160501.hdf','sample_C_20160501.hdf']
    )

    artifact_source = (
        Hdf5Source(
            'sample_ABC_padded_20160501.defects.hdf',
            datasets = {
                ArrayKeys.RAW: 'defect_sections/raw',
                ArrayKeys.ALPHA_MASK: 'defect_sections/mask',
            },
            array_specs = {
                ArrayKeys.RAW: ArraySpec(voxel_size=(40, 4, 4)),
                ArrayKeys.ALPHA_MASK: ArraySpec(voxel_size=(40, 4, 4)),
            }
        ) +
        RandomLocation(min_masked=0.05, mask_array_type=ArrayKeys.ALPHA_MASK) +
        Snapshot(
            {
                ArrayKeys.RAW: 'volumes/raw',
            },
            every=1,
            output_filename='defect_{id}.hdf') +
        Normalize() +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        ElasticAugment(
            [4,40,40],
            [0,2,2],
            [0,math.pi/2.0],
            subsample=8) +
        SimpleAugment(transpose_only_xy=True)
    )

    batch_provider_tree = (
        data_sources +
        RandomProvider() +
        ExcludeLabels([8094], ignore_mask_erode=12) +
        SplitAndRenumberSegmentationLabels() +
        ElasticAugment(
            [4,40,40],
            [0,2,2],
            [0,math.pi/2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=25,
            subsample=8) +
        SimpleAugment(transpose_only_xy=True) +
        GrowBoundary(steps=3, only_xy=True) +
        DownSample(
            {
                ArrayKeys.GT_LABELS_2: (2, ArrayKeys.GT_LABELS),
                ArrayKeys.GT_LABELS_4: (4, ArrayKeys.GT_LABELS)
            }
        ) +
        AddGtAffinities(affinity_neighborhood) +
        AddBoundaryDistanceGradients(
            gradient_array_type=ArrayKeys.GT_BOUNDARY_GRADIENT,
            distance_array_type=ArrayKeys.GT_BOUNDARY_DISTANCE,
            boundary_array_type=ArrayKeys.GT_BOUNDARY,
            normalize='l2') +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        DefectAugment(
            prob_missing=0.03,
            prob_low_contrast=0.01,
            prob_artifact=0.03,
            artifact_source=artifact_source,
            contrast_scale=0.1) +
        ZeroOutConstSections() +
        BalanceLabels({ArrayKeys.GT_AFFINITIES: ArrayKeys.LOSS_SCALE}) +
        PreCache(
            cache_size=10,
            num_workers=5) +
        Snapshot(
            {
                ArrayKeys.RAW: 'volumes/raw',
                ArrayKeys.GT_LABELS: 'volumes/labels/neuron_ids',
                ArrayKeys.GT_LABELS_2: 'volumes/labels/neuron_ids_2',
                ArrayKeys.GT_LABELS_4: 'volumes/labels/neuron_ids_4',
                ArrayKeys.GT_IGNORE: 'volumes/labels/mask',
                ArrayKeys.GT_AFFINITIES: 'volumes/labels/affinities',
                ArrayKeys.GT_BOUNDARY_GRADIENT:
                    'volumes/labels/boundary_gradient',
                ArrayKeys.GT_BOUNDARY_DISTANCE:
                    'volumes/labels/boundary_distance',
                ArrayKeys.GT_BOUNDARY:
                    'volumes/labels/boundary',
            },
            every=1,
            output_filename='final_it={iteration}_id={id}.hdf') +
        PrintProfilingStats(every=n)
    )

    print("Requesting", n, "batches")

    with build(batch_provider_tree):
        for i in range(n):
            batch_provider_tree.request_batch(request)

    print("Finished")


if __name__ == "__main__":
    train()
