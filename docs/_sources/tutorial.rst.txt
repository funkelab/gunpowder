.. _sec_tutorial:

Tutorial: boundary prediction for instance segmentation
=======================================================

(written by Sherry Ding)

.. automodule:: gunpowder

This is a tutorial about how to use ``gunpowder`` to assemble pipelines to train and test a neural network. As an 
example, here we do neuron segmentation on the `cremi dataset <https://cremi.org/data/>`_ using a *3D U-Net*. To segment 
neurons, we predict inter-voxel affinities from volumes of raw data. A ``gunpowder`` processing pipeline consists 
of nodes that can be chained together using **+**. The major nodes of our training pipeline in this example includes 
reading in sources, data augmentation, processing the labels, classifier training, and batch saving. Now we'll describe 
the `code <https://github.com/funkey/gunpowder/tree/release-v1.0/examples/cremi>`_ in parts.

Before we start, packages *malis* and *tensorflow* need to be installed for this example.

.. code-block:: bash

    sudo apt-get install libboost-all-dev gcc  # necessary for package malis (example for Ubuntu)
    pip install malis
    conda install tensorflow-gpu


Create a network to train with
------------------------------

Before assembling a pipeline and training on the data, it is required to build the neural network first, as it will be 
called in the pipeline to train with. In this example, we build a *3D U-Net* with *tensorflow* as our neural network, and 
store it in a *meta* file with its configuration in a *json* file for calling in the pipeline. The following script creates 
a network for training and a larger network for faster prediction/testing.

.. code-block:: bash

    python mknet.py 


Assemble the training pipeline
------------------------------

Now we can assemble the training pipeline using ``gunpowder``.

.. code-block:: python

    from __future__ import print_function
    import gunpowder as gp
    import json
    import math
    import logging

    logging.basicConfig(level=logging.INFO)

First of all, we have to create :class:`ArrayKeys<ArrayKey>` for all arrays that the pipeline will use, i.e., give names to 
our :class:`Arrays<Array>` for later requests or access. We create :class:`ArrayKeys<ArrayKey>` for raw intensities, ground 
truth labels with unique IDs, ground truth affinities, weight to use to balance the loss, predicted affinities, and gredient 
of the loss with regard to the predicted affinities.

.. code-block:: python

    # raw intensities
    raw = gp.ArrayKey('RAW')

    # objects labeled with unique IDs
    gt_labels = gp.ArrayKey('LABELS')

    # array of per-voxel affinities to direct neighbors
    gt_affs= gp.ArrayKey('AFFINITIES')

    # weights to use to balance the loss
    loss_weights = gp.ArrayKey('LOSS_WEIGHTS')

    # the predicted affinities
    pred_affs = gp.ArrayKey('PRED_AFFS')

    # the gredient of the loss wrt to the predicted affinities
    pred_affs_gradients = gp.ArrayKey('PRED_AFFS_GRADIENTS')

Next, we load the *3D U-Net* that we built and saved for training, and use its configurations to set the input and output size.

.. code-block:: python

    with open('train_net_config.json', 'r') as f:
        net_config = json.load(f)

    # get the input and output size in world units (nm, in this case)
    voxel_size = gp.Coordinate((40, 4, 4))
    input_size = gp.Coordinate(net_config['input_shape'])*voxel_size
    output_size = gp.Coordinate(net_config['output_shape'])*voxel_size

We use :class:`BatchRequest<BatchRequest>` to formulate the request for what a batch should contain. For training, a batch 
should contain raw data, ground truth affinities, and loss weights. In this example, we also request a batch for snapshot. 
:class:`Snapshot<Snapshot>` saves a passing batch in an *hdf* file for inspection.

.. code-block:: python

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

Now we are going to assemble the training pipeline. As mentioned before, a ``gunpowder`` pipeline consists of nodes that are 
chained using **+**. In this example, the first node deals with the sources.

.. code-block:: python

    pipeline = (

        # a tuple of sources, one for each sample (A, B, and C) provided by the CREMI challenge
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

Here, :class:`Hdf5Source<Hdf5Source>` provides arrays from samples that are in HDF5 format. :class:`Normalize<Normalize>` 
normalizes values of the array to be floats between 0 and 1. :class:`RandomLocation<RandomLocation>` chooses a random 
loacation for each request batch.

Next we choose a random source, and do data augmentation. TODO: random source. Data augmentation "increases" the amount 
of training data by augmenting them via a number of random transformations. The augmentations we use here are elastic 
augmentation, transpose and mirror augmentation, as well as scaling and shifting the intensity. 

.. code-block:: python

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

We also need to process the ground truth labels and affinities, e.g., grow a boundary between labels, convert labels 
into affinities, and balance samples.

.. code-block:: python

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

In this example, as our batch requests are repeatly the same, we pre-cache batches. :class:`PreCache<PreCache>` 
means that a set of workers is spawned to pre-cache the batches in parallel processes for serving subsequent 
requests quickly.

.. code-block:: python

        # pre-cache batches from the point upstream
        gp.PreCache(
            cache_size=10,
            num_workers=5) +

The next node performs one training iteration for each passing batch.

.. code-block:: python

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

We save the passing batch using :class:`Snapshot<Snapshot>` and show a summary of time consuming.  

.. code-block:: python

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

The final thing we need to do is requesting batches for the pipeline.

.. code-block:: python

    print("Training for", iterations, "iterations")

    with gp.build(pipeline):
        for i in range(iterations):
            pipeline.request_batch(request)

    print("Finished")

Let's put all above codes into a function called *train*, with an input parameter *iterations*. One iteration means 
training on one batch once.

.. code-block:: python

    def train(iterations):
        ...

In this example, we repeatly request a batch and train on it for 200000 times.

.. code-block:: python

    if __name__ == "__main__":
        train(200000)


Assemble the testing pipeline
-----------------------------

After trained the network, we also use ``gunpowder`` to assemble the testing pipeline.

.. code-block:: python

    from __future__ import print_function
    import gunpowder as gp
    import json

The first is still creating :class:`ArrayKeys<ArrayKey>` for all arrays. We create :class:`ArrayKeys<ArrayKey>` 
for raw intensities of testing data and predicted affinities.

.. code-block:: python

    # raw intensities
    raw = gp.ArrayKey('RAW')

    # the predicted affinities
    pred_affs = gp.ArrayKey('PRED_AFFS')

Load the *3D U-Net* that we built and saved for testing, and use its configurations to set the input and output size.

.. code-block:: python

    with open('test_net_config.json', 'r') as f:
        net_config = json.load(f)

    # get the input and output size in world units (nm, in this case)
    voxel_size = gp.Coordinate((40, 4, 4))
    input_size = gp.Coordinate(net_config['input_shape'])*voxel_size
    output_size = gp.Coordinate(net_config['output_shape'])*voxel_size
    context = input_size - output_size

We formulate the request for what a batch should contain. For testing, a batch should contain raw data and predicted 
affinities.

.. code-block:: python

    # formulate the request for what a batch should contain
    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(pred_affs, output_size)

Next is to assemble the testing pipeline. The pipeline for testing/prediction is much simpler. It should at least 
include a source node and a prediction node.

.. code-block:: python

    source = gp.Hdf5Source(
        'sample_A_padded_20160501.hdf',
        datasets = {
            raw: 'volumes/raw'
        })

    # get the ROI provided for raw (we need it later to calculate the ROI in
    # which we can make predictions)
    with gp.build(source):
        raw_roi = source.spec[raw].roi

    pipeline = (

        # read from HDF5 file
        source +

        # convert raw to float in [0, 1]
        gp.Normalize(raw) +

        # perform one training iteration for each passing batch (here we use
        # the tensor names earlier stored in train_net.config)
        gp.tensorflow.Predict(
            graph='test_net.meta',
            checkpoint='train_net_checkpoint_%d'%iteration,
            inputs={
                net_config['raw']: raw
            },
            outputs={
                net_config['pred_affs']: pred_affs
            },
            array_specs={
                pred_affs: gp.ArraySpec(roi=raw_roi.grow(-context, -context))
            }) +

We also contain a :class:`Hdf5Write<Hdf5Write>` node to store all passing batches, and a node showing a summary 
of time consuming.

.. code-block:: python

        # store all passing batches in the same HDF5 file
        gp.Hdf5Write(
            {
                raw: '/volumes/raw',
                pred_affs: '/volumes/pred_affs',
            },
            output_filename='predictions_sample_A.hdf',
            compression_type='gzip'
        ) +

        # show a summary of time spend in each node every 10 iterations
        gp.PrintProfilingStats(every=10) +

Last :class:`Scan<Scan>` node is used to iteratively request batches over the whole dataset in a scanning fashion

.. code-block:: python

        # iterate over the whole dataset in a scanning fashion, emitting
        # requests that match the size of the network
        gp.Scan(reference=request)
    )

Finally, we request an empty batch from :class:`Scan<Scan>` to trigger scanning of the dataset.

.. code-block:: python

    with gp.build(pipeline):
        # request an empty batch from Scan to trigger scanning of the dataset
        # without keeping the complete dataset in memory
        pipeline.request_batch(gp.BatchRequest())
