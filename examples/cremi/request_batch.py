from __future__ import print_function

import math
import numpy as np
import random

from gunpowder import *

def train():

    random.seed(42)
    set_verbose()

    affinity_neighborhood = np.array([[-1,0,0],[0,-1,0],[0,0,-1]])
    raw = ArrayKey('RAW')
    labels = ArrayKey('GT_LABELS')
    labels_2 = ArrayKey('GT_LABELS_2')
    labels_4 = ArrayKey('GT_LABELS_4')
    ignore = ArrayKey('GT_IGNORE')
    affinities = ArrayKey('GT_AFFINITIES')
    boundary = ArrayKey('GT_BOUNDARY')
    boundary_gradient = ArrayKey('GT_BOUNDARY_GRADIENT')
    boundary_distance = ArrayKey('GT_BOUNDARY_DISTANCE')
    loss_scale = ArrayKey('LOSS_SCALE')
    alpha_mask = ArrayKey('ALPHA_MASK')
    n = 35

    request = BatchRequest()
    request.add(raw, Coordinate((84,268,268))*(40,4,4))
    request.add(labels, Coordinate((56,56,56))*(40,4,4))
    request.add(labels_2, Coordinate((56,56,56))*(40,4,4))
    request.add(labels_4, Coordinate((56,56,56))*(40,4,4))
    request.add(ignore, Coordinate((56,56,56))*(40,4,4))
    request.add(affinities, Coordinate((56,56,56))*(40,4,4))
    request.add(boundary_gradient, Coordinate((56,56,56))*(40,4,4))
    request.add(boundary_distance, Coordinate((56,56,56))*(40,4,4))
    request.add(boundary, Coordinate((56,56,56))*(40,4,4))
    request.add(loss_scale, Coordinate((56,56,56))*(40,4,4))

    data_sources = tuple(
        Hdf5Source(
            sample,
            datasets = {
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
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
                raw: 'defect_sections/raw',
                alpha_mask: 'defect_sections/mask',
            },
            array_specs = {
                raw: ArraySpec(voxel_size=(40, 4, 4)),
                alpha_mask: ArraySpec(voxel_size=(40, 4, 4)),
            }
        ) +
        RandomLocation(min_masked=0.05, mask_array_key=alpha_mask) +
        Snapshot(
            {
                raw: 'volumes/raw',
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
                labels_2: (2, labels),
                labels_4: (4, labels)
            }
        ) +
        AddGtAffinities(affinity_neighborhood) +
        AddBoundaryDistanceGradients(
            gradient_array_key=boundary_gradient,
            distance_array_key=boundary_distance,
            boundary_array_key=boundary,
            normalize='l2') +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        DefectAugment(
            prob_missing=0.03,
            prob_low_contrast=0.01,
            prob_artifact=0.03,
            artifact_source=artifact_source,
            contrast_scale=0.1) +
        ZeroOutConstSections() +
        BalanceLabels({affinities: loss_scale}) +
        PreCache(
            cache_size=10,
            num_workers=5) +
        Snapshot(
            {
                raw: 'volumes/raw',
                labels: 'volumes/labels/neuron_ids',
                labels_2: 'volumes/labels/neuron_ids_2',
                labels_4: 'volumes/labels/neuron_ids_4',
                ignore: 'volumes/labels/mask',
                affinities: 'volumes/labels/affinities',
                boundary_gradient: 'volumes/labels/boundary_gradient',
                boundary_distance: 'volumes/labels/boundary_distance',
                boundary: 'volumes/labels/boundary',
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
