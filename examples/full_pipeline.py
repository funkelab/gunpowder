from gunpowder import *
import math
import time
import random
import malis

random.seed(42)

affinity_neighborhood = malis.mknhood3d()
print(affinity_neighborhood)

solver_parameters = SolverParameters()
solver_parameters.train_net = 'net_train_euclid.prototxt'
solver_parameters.base_lr = 0.00005
solver_parameters.momentum = 0.99
solver_parameters.weight_decay = 0.000005
solver_parameters.lr_policy = 'inv'
solver_parameters.gamma = 0.0001
solver_parameters.power = 0.75
solver_parameters.max_iter = 6000
solver_parameters.snapshot = 2000
solver_parameters.snapshot_prefix = 'net'
solver_parameters.type = 'Adam'
solver_parameters.display = 1
solver_parameters.train_state.add_stage('euclid')

# simulate many sources (here we point to the same file always)
sources = tuple(
    Hdf5Source(
            'test2.hdf',
            raw_dataset='volumes/raw',
            gt_dataset='volumes/labels/neuron_ids',
            gt_mask_dataset='volumes/labels/mask') +\
    Normalize() +\
    RandomLocation()
    for i in range(10)
)

# create a batch provider by concatenation of filters
batch_provider = (
        sources +
        RandomProvider() +
        ExcludeLabels([416759, 397008], 8) +
        Reject() +
        Snapshot(every=1, output_dir='snapshots_original') +
        ElasticAugmentation([4,40,40], [0,2,2], [0,math.pi/2.0]) +
        SimpleAugment(transpose_only_xy=True) +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        DefectAugment(prob_missing=0.1, prob_low_contrast=0.1, contrast_scale=0.1) +
        CropGt(1) +
        GrowBoundary(steps=3, only_xy=True) +
        AddGtAffinities(affinity_neighborhood) +
        CropGt() +
        Snapshot(every=1) +
        IntensityScaleShift(2, -1) +
        ZeroOutConstSections() +
        PreCache(
                lambda : BatchSpec(
                        (84,268,268),
                        (56,56,56),
                        with_gt=True,
                        with_gt_mask=True,
                        with_gt_affinities=True),
                cache_size=20,
                num_workers=10) +
        Train(solver_parameters, dry_run=True) +
        Snapshot(every=1, output_dir='snapshots_final')
)

n = 10
print("Training for " + str(n) + " iterations")
batch_provider.initialize_all()
for i in range(n):
    batch_provider.request_batch(None)
