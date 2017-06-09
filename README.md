gunpowder
=========

A data-loading, training, and processing DAG for greentea.

Based on [`PyGreentea`](https://github.com/TuragaLab/PyGreentea) by William Grisaitis, Fabian Tschopp, and Srini Turaga.

First steps
-----------

In `gunpowder`, you assemble a training pipeline as a directed acyclic graph
(DAG) of batch providers. Everything starts with a "source", a batch provider
with no inputs:

```python
source =
    Hdf5Source(
            'example.hdf',
            raw_dataset='volumes/raw',
            gt_dataset='volumes/labels/neuron_ids',
            gt_mask_dataset='volumes/labels/mask')
```

Every batch provider can be asked for a batch via a request (e.g., shape,
offset, which kind of volumes to provide). Starting from a source, you can
chain batch providers to build a DAG. When a non-source batch provider is asked
for a batch, it passes the request on upstream (i.e., towards a source) to
receive a batch, possibly modifies it, and passes the batch on downstream.

As an example, this scheme allows the modelling of on-the-fly data augmentation
as a batch provider:
```python
augment =
    ElasticAugment(
        control_point_spacing=[4,40,40],
        jitter_sigma=[0,2,2],
        rotation_interval=[0,math.pi/2.0])
```
When `augment` is asked for a batch via a request, the request is automatically
changed to request an upstream batch large enough to perform the elastic
augmentation seamlessly.

Another example is the random selection of locations inside a source:
```python
random =
    RandomLocation()
```

Training itself is modelled as a batch provider. It takes a batch, performs one
training iteration, and adds the current prediction and loss to the batch:

```python
solver_parameters = SolverParameters()
# set solver parameters (network, learning rate, optimizer, etc.)
train =
    Train(solver_parameters, use_gpu=0)
```

Putting it together, a very simple pipeline for training 1000 iterations would be
```python
pipeline = source + random + augment + train

request = BatchRequest()
request.add_volume_request(VolumeType.RAW, (84,268,268))
request.add_volume_request(VolumeType.GT_LABELS, (56,56,56))
request.add_volume_request(VolumeType.GT_MASK, (56,56,56))

with build(pipeline) as p:
  for i in range(1000):
    p.request_batch(request)
```
Note that we use a `BatchRequest` object to communicate downstream the
requirements for a batch. In the example, we are interested in batches of
certain sizes (fitting the network we want to train) with raw data,
ground-truth labels, and a mask.

For a full working example (with many more batch providers), see [the example
directory](examples/cremi/).

Writing Custom Batch Providers
------------------------------

The simplest batch provider is a `BatchFilter`, which has only one upstream
provider. To create a new one, subclass it and override `prepare` and/or
`process`:

```python
class ExampleFilter(BatchFilter):

  def prepare(self, request):
    pass

  def process(self, batch, request):
    pass
```

`prepare` and `process` will be called in an alternating fashion. `prepare` is
called first, when a `BatchRequest` is passed upstream through the filter. Your
filter is free to change the request in any way it needs to, for example, by
increasing the requested sizes. After `prepare`, `process` will be called with
a batch going downstream, which is the upstream's response to the request you
modified in `prepare`. In `process`, your filter should make all necessary
changes to the batch and ensure it meets the original downstream request
earlier communicated to `prepare` (given as `request` parameter in `process`
for convenience).

For an example of a batch filter changing both the spec going upstream and the
batch going downstream, see
[ElasticAugment](gunpowder/elastic_augment.py).
