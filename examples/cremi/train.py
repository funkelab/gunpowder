from __future__ import print_function
import gunpowder as gp
import json
import math

def train(iterations):

    ##################
    # DECLARE ARRAYS #
    ##################

    # raw intensities
    raw = gp.ArrayKey('RAW')

    # objects labelled with unique IDs
    gt_labels = gp.ArrayKey('LABELS')

    # array of per-voxel affinities to direct neighbors
    gt_affs= gp.ArrayKey('AFFINITIES')

    # weights to use to balance the loss
    loss_weights = gp.ArrayKey('LOSS_WEIGHTS')

    # the predicted affinities
    pred_affs = gp.ArrayKey('PRED_AFFS')

    # the gredient of the loss wrt to the predicted affinities
    pred_affs_gradients = gp.ArrayKey('PRED_AFFS_GRADIENTS')

    ####################
    # DECLARE REQUESTS #
    ####################

    with open('train_net_config.json', 'r') as f:
        net_config = json.load(f)

    # get the input and output size in world units (nm, in this case)
    voxel_size = gp.Coordinate((40, 4, 4))
    input_size = gp.Coordinate(net_config['input_shape'])*voxel_size
    output_size = gp.Coordinate(net_config['output_shape'])*voxel_size

    # formulate the request for what a batch should (at least) contain
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(gt_affs, output_size)
    request.add(loss_weights, output_size)

    # when we make a snapshot for inspection (see below), we also want to
    # request the predicted affinities and gradients of the loss wrt the
    # affinities
    snapshot_request = gp.BatchRequest()
    snapshot_request[pred_affs] = request[gt_affs]
    snapshot_request[pred_affs_gradients] = request[gt_affs]

    ##############################
    # ASSEMBLE TRAINING PIPELINE #
    ##############################

    pipeline = (

        # a tuple of sources, one for each sample (A, B, and C) provided by the
        # CREMI challenge
        tuple(

            # read batches from the HDF5 file
            gp.Hdf5Source(
                'sample_'+s+'_padded_20160501.hdf',
                datasets = {
                    raw: 'volumes/raw',
                    gt_labels: 'volumes/labels/neuron_ids'
                }
            ) +

            # convert raw to float in [0, 1]
            gp.Normalize(raw) +

            # chose a random location for each requested batch
            gp.RandomLocation()

            for s in ['A', 'B', 'C']
        ) +

        # chose a random source (i.e., sample) from the above
        gp.RandomProvider() +

        # elastically deform the batch
        gp.ElasticAugment(
            [4,40,40],
            [0,2,2],
            [0,math.pi/2.0],
            prob_slip=0.05,
            prob_shift=0.05,
            max_misalign=25) +

        # apply transpose and mirror augmentations
        gp.SimpleAugment(transpose_only=[1, 2]) +

        # scale and shift the intensity of the raw array
        gp.IntensityAugment(
            raw,
            scale_min=0.9,
            scale_max=1.1,
            shift_min=-0.1,
            shift_max=0.1,
            z_section_wise=True) +

        # grow a boundary between labels
        gp.GrowBoundary(
            gt_labels,
            steps=3,
            only_xy=True) +

        # convert labels into affinities between voxels
        gp.AddAffinities(
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            gt_labels,
            gt_affs) +

        # create a weight array that balances positive and negative samples in
        # the affinity array
        gp.BalanceLabels(
            gt_affs,
            loss_weights) +

        # pre-cache batches from the point upstream
        gp.PreCache(
            cache_size=10,
            num_workers=5) +

        # perform one training iteration for each passing batch (here we use
        # the tensor names earlier stored in train_net.config)
        gp.tensorflow.Train(
            'train_net',
            net_config['optimizer'],
            net_config['loss'],
            inputs={
                net_config['raw']: raw,
                net_config['gt_affs']: gt_affs,
                net_config['loss_weights']: loss_weights
            },
            outputs={
                net_config['pred_affs']: pred_affs
            },
            gradients={
                net_config['pred_affs']: pred_affs_gradients
            },
            save_every=1) +

        # save the passing batch as an HDF5 file for inspection
        gp.Snapshot(
            {
                raw: '/volumes/raw',
                gt_labels: '/volumes/labels/neuron_ids',
                gt_affs: '/volumes/labels/affs',
                pred_affs: '/volumes/pred_affs',
                pred_affs_gradients: '/volumes/pred_affs_gradients'
            },
            output_dir='snapshots',
            output_filename='batch_{iteration}.hdf',
            every=100,
            additional_request=snapshot_request,
            compression_type='gzip') +

        # show a summary of time spend in each node every 10 iterations
        gp.PrintProfilingStats(every=10)
    )

    #########
    # TRAIN #
    #########

    print("Training for", iterations, "iterations")

    with gp.build(pipeline):
        for i in range(iterations):
            pipeline.request_batch(request)

    print("Finished")

if __name__ == "__main__":
    train(200000)
