from __future__ import print_function
import sys
from gunpowder import *
from gunpowder.caffe import *
import malis
import glob
import math

# the training HDF files
samples = [
    'trvol-250-1.hdf',
    # add more here
]

# after how many iterations to switch from Euclidean loss to MALIS
phase_switch = 10000

def train_until(max_iteration, gpu):
    '''Resume training from the last stored network weights and train until ``max_iteration``.'''

    set_verbose(False)

    # get most recent training result
    solverstates = [ int(f.split('.')[0].split('_')[-1]) for f in glob.glob('net_iter_*.solverstate') ]
    if len(solverstates) > 0:
        trained_until = max(solverstates)
        print("Resuming training from iteration " + str(trained_until))
    else:
        trained_until = 0
        print("Starting fresh training")

    if trained_until < phase_switch and max_iteration > phase_switch:
        # phase switch lies in-between, split training into to parts
        train_until(phase_switch, gpu)
        trained_until = phase_switch

    if max_iteration <= phase_switch:
        phase = 'euclid'
    else:
        phase = 'malis'
    print("Traing until " + str(max_iteration) + " in phase " + phase)

    # setup training solver and network
    solver_parameters = SolverParameters()
    solver_parameters.train_net = 'net.prototxt'
    solver_parameters.base_lr = 0.5e-4
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
    if trained_until > 0:
        solver_parameters.resume_from = 'net_iter_' + str(trained_until) + '.solverstate'
    else:
        solver_parameters.resume_from = None
    solver_parameters.train_state.add_stage(phase)

    # input and output shapes of the network, needed to formulate matching batch 
    # requests
    input_shape = (196,)*3
    output_shape = (92,)*3

    # volumes to request for each batch
    request = BatchRequest()
    request.add_volume_request(VolumeTypes.RAW, input_shape)
    request.add_volume_request(VolumeTypes.GT_LABELS, output_shape)
    request.add_volume_request(VolumeTypes.GT_MASK, output_shape)
    request.add_volume_request(VolumeTypes.GT_AFFINITIES, output_shape)
    if phase == 'euclid':
        request.add_volume_request(VolumeTypes.LOSS_SCALE, output_shape)

    # create a tuple of data sources, one for each HDF file
    data_sources = tuple(

        # provide volumes from the given HDF datasets
        Hdf5Source(
            sample,
            datasets = {
                VolumeTypes.RAW: 'volumes/raw',
                VolumeTypes.GT_LABELS: 'volumes/labels/neuron_ids',
                VolumeTypes.GT_MASK: 'volumes/labels/mask',
            }
        ) +

        # ensure RAW is in float in [0,1]
        Normalize() +

        # zero-pad provided RAW and GT_MASK to be able to draw batches close to 
        # the boundary of the available data
        Pad(
            {
                VolumeTypes.RAW: Coordinate((100, 100, 100)),
                VolumeTypes.GT_MASK: Coordinate((100, 100, 100))
            }
        ) +

        # chose a random location inside the provided volumes
        RandomLocation() +

        # reject batches wich do contain less than 50% labelled data
        Reject()

        for sample in samples
    )

    # attach data sources to training pipeline
    train_pipeline = (

        data_sources +

        # randomly select any of the data sources
        RandomProvider() +

        # elastically deform and rotate
        ElasticAugment([40,40,40], [2,2,2], [0,math.pi/2.0], prob_slip=0.01, max_misalign=1, subsample=8) +

        # randomly mirror and transpose
        SimpleAugment() +

        # grow a 0-boundary between labelled objects
        GrowBoundary(steps=4) +

        # relabel connected label components inside the batch
        SplitAndRenumberSegmentationLabels() +

        # compute ground-truth affinities from labels
        AddGtAffinities(malis.mknhood3d()) +

        # add a LOSS_SCALE volume to balance positive and negative classes for 
        # Euclidean training
        BalanceAffinityLabels() +

        # randomly scale and shift intensities
        IntensityAugment(0.9, 1.1, -0.1, 0.1) +

        # ensure RAW is in [-1,1]
        IntensityScaleShift(2,-1) +

        # use 10 workers to pre-cache batches of the above pipeline
        PreCache(
            cache_size=40,
            num_workers=10) +

        # perform one training iteration
        Train(
            solver_parameters,
            inputs={
                VolumeTypes.RAW: 'data',
                VolumeTypes.GT_AFFINITIES: 'aff_label',
            },
            outputs={
                VolumeTypes.PRED_AFFINITIES: 'aff_pred'
            },
            gradients={
                VolumeTypes.LOSS_GRADIENT: 'aff_pred'
            },
            resolutions={
                VolumeTypes.PRED_AFFINITIES: Coordinate((8,8,8)),
                VolumeTypes.LOSS_GRADIENT: Coordinate((8,8,8)),
            },
            use_gpu=gpu) +

        # save every 100th batch into an HDF5 file for manual inspection
        Snapshot(
            every=100,
            output_filename='batch_{iteration}.hdf',
            additional_request=BatchRequest({VolumeTypes.LOSS_GRADIENT: request.volumes[VolumeTypes.GT_AFFINITIES]})) +

        # add useful profiling stats to identify bottlenecks
        PrintProfilingStats(every=10)
    )

    iterations = max_iteration - trained_until
    assert iterations >= 0

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(iterations):
            b.request_batch(request)
    print("Training finished")

if __name__ == "__main__":
    iteration = int(sys.argv[1])
    gpu = int(sys.argv[2])
    train_until(iteration, gpu)
