gunpowder
=========

A data-loading and training DAG for greentea.

Usage
-----

In `gunpowder`, you assemble a training pipeline as a directed acyclic graph
(DAG) of batch providers. Everything starts with a "source", a batch provider
with no inputs, i.e., a leaf in the DAG.

```
source =
    Hdf5Source(
            'example.hdf',
            raw_dataset='volumes/raw',
            gt_dataset='volumes/labels/neuron_ids',
            gt_mask_dataset='volumes/labels/mask')
```

Batch providers can be asked for a batch by providing a (possibly partial)
specification (e.g., shape, offset, which kind of ground-truth to provide).

In the DAG, batch specifications flow upstream, and batches downstream.
Starting from a source, you can add downstream batch providers to modify the
batch in various ways.

One use-case is on-the-fly data augmentation, e.g.:
```
augment =
    ElasticAugmentation(
        control_point_spacing=[4,40,40],
        jitter_sigma=[0,2,2],
        rotation_interval=[0,math.pi/2.0])
```

Another example is random selection of locations inside a source:
```
random =
    RandomLocation()
```

Training itself is modelled as a batch provider. It takes a batch, performs one
training iteration, and adds the current prediction and loss to the batch:

```
solver_parameters = SolverParameters()
# set solver parameters (network, learning rate, optimizer, etc.)
train =
    Train(solver_parameters, use_gpu=0)
```

Putting it together, a very simple pipeline for training 1000 iterations would be
```
pipeline = source + random + augment + train
pipeline.initialize_all()

for i in range(1000):
  pipeline.request_batch(
      BatchSpec(
          input_shape=(84,268,268),
          output_shape=(56,56,56),
          with_gt=True,
          with_gt_mask=True,
          with_gt_affinities=True))
```
Note that we use a `BatchSpec` object to communicate downstream the
requirements for a batch. In the example, we are interested in batches of
certain sizes (to fit the network we want to train) with ground-truth labels,
mask, and affinities.
