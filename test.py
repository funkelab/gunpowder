import gunpowder
import math
import time
import random
import malis

random.seed(42)

affinity_neighborhood = malis.mknhood3d()
print(affinity_neighborhood)

# simulate many sources (here we point to the same file always)
sources = tuple(
    gunpowder.Hdf5Source(
            'test2.hdf',
            raw_dataset='volumes/raw',
            gt_dataset='volumes/labels/neuron_ids',
            gt_mask_dataset='volumes/labels/mask') +\
    gunpowder.RandomLocation()
    for i in range(10)
)

# create a batch provider by concatenation of filters
batch_provider =\
        sources +\
        gunpowder.RandomProvider() +\
        gunpowder.Snapshot(every=1, output_dir='snapshots_original') +\
        gunpowder.ExcludeLabels([416759, 397008], 8) +\
        gunpowder.Reject() +\
        gunpowder.ElasticAugmentation([1,20,20], [0,2,2], [0,math.pi/2.0]) +\
        gunpowder.SimpleAugment(transpose_only_xy=True) +\
        gunpowder.DefectAugment(prob_missing=0.1, prob_low_contrast=0.1, contrast_scale=0.1) +\
        gunpowder.CropGt(1) +\
        gunpowder.GrowBoundary(steps=3, only_xy=True) +\
        gunpowder.AddGtAffinities(affinity_neighborhood) +\
        gunpowder.CropGt() +\
        gunpowder.Snapshot(every=1) +\
        gunpowder.PreCache(
                lambda : gunpowder.BatchSpec(
                        (84,268,268),
                        (56,56,56),
                        with_gt=True,
                        with_gt_mask=True,
                        with_gt_affinities=True),
                cache_size=10,
                num_workers=2)

print("Trying to get a batch...")

batch_provider.initialize_all()
batch = batch_provider.request_batch(None)

# print("Starting training...")

# solver_parameters = gunpowder.SolverParameters()
# solver_parameters.train_net = 'net_train_euclid.prototxt'
# solver_parameters.base_lr = 0.00005
# solver_parameters.momentum = 0.99
# solver_parameters.weight_decay = 0.000005
# solver_parameters.lr_policy = 'inv'
# solver_parameters.gamma = 0.0001
# solver_parameters.power = 0.75
# solver_parameters.max_iter = 6000
# solver_parameters.snapshot = 2000
# solver_parameters.snapshot_prefix = 'net'
# solver_parameters.type = 'Adam'
# solver_parameters.display = 1
# solver_parameters.train_state.add_stage('euclid')

# use_gpu = None
# solver = gunpowder.init_solver(solver_parameters, use_gpu)

# gunpowder.train(solver, batch_provider, use_gpu)
