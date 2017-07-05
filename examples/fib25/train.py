from __future__ import print_function

import malis

from gunpowder import *
from gunpowder.caffe import Train, SolverParameters

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
    solver_parameters.snapshot = 10
    solver_parameters.snapshot_prefix = 'net'
    solver_parameters.type = 'Adam'
    solver_parameters.resume_from = None
    solver_parameters.train_state.add_stage('euclid')

    batch_spec = BatchSpec(
        (84,268,268),
        (56,56,56),
        with_gt=True,
        with_gt_mask=False,
        with_gt_affinities=True
    )

    # create a batch provider by concatenation of filters
    batch_provider = (
        DvidSource(
            hostname='slowpoke3',
            port=32788,
            uuid='341',
            raw_array_name='grayscale',
            gt_array_name='groundtruth_pruned',
        ) +
        Normalize() +
        RandomLocation() +
        AddGtAffinities(affinity_neighborhood) + 
        BalanceAffinityLabels() +
        PreCache(
            lambda: batch_spec,
            cache_size=3,
            num_workers=2
        ) +
        Train(solver_parameters, use_gpu=0) +
        Snapshot(every=1)
    )

    n = 20
    print("Training for", n, "iterations")

    with build(batch_provider) as minibatch_maker:
        for i in range(n):
            minibatch_maker.request_batch(batch_spec)

    print("Finished")


if __name__ == "__main__":
    train()
