from __future__ import print_function

import math
import random
import time

import malis

import gunpowder
from gunpowder import caffe
from gunpowder import *


random.seed(42)

def train():
    set_verbose()
    affinity_neighborhood = malis.mknhood3d()
    solver_paramaters = caffe.SolverParameters()
    solver_paramaters.train_net = 'net.prototxt'
    print(affinity_neighborhood)
    data_source_trees = tuple(
        Hdf5Source(
            'sample_A_20160501.hdf',
            raw_dataset='volumes/raw',
            gt_dataset='volumes/labels/neuron_ids'
        ) +
        Normalize() +
        RandomLocation()
        for i in range(10)
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
        Train(solver_paramaters, use_gpu=0) +
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
