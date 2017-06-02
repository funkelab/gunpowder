from __future__ import print_function

import math

from gunpowder.ext import malis
from gunpowder import *
from gunpowder.caffe import *

def train():

    set_verbose()

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
        ElasticAugment([4,40,40], [0,2,2], [0,math.pi/2.0]) +
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
        Train(solver_parameters, use_gpu=0) +
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
