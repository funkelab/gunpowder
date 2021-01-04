.. _sec_first_steps:

First steps
===========

.. automodule:: gunpowder

Declaring arrays
----------------

Before you start assembling a training of prediction pipeline, you have to
create :class:`ArrayKeys<ArrayKey>` for all arrays your pipeline will use.
These keys are used later to formulate a request for an array or to access the
actual array associated with that key.

In the example here, we assume we have a raw dataset, together with
ground-truth labels and a mask which lets us know where ground-truth is
available.

.. code-block:: python

  import gunpowder as gp

  raw = gp.ArrayKey('RAW')
  gt = gp.ArrayKey('GT')
  gt_mask = gp.ArrayKey('MASK')


Creating a source
-----------------

In ``gunpowder``, you assemble a training pipeline as a directed acyclic
graph (DAG) of :class:`BatchProvider<BatchProvider>`. The leaves of
your DAG are called sources, i.e., batch providers with no inputs:

.. code-block:: python

  source =
      gp.Hdf5Source(
          'example.hdf',
          {
              raw: 'volumes/raw',
              gt: 'volumes/labels/neuron_ids',
              gt_mask_dataset: 'volumes/labels/mask'
          }
      )

Chaining batch providers
------------------------

Every batch provider can be asked for a batch via a :class:`BatchRequest`
(e.g., shape, offset, which kind of volumes to provide) to provide a
:class:`Batch`. Starting from one or multiple sources, you can chain batch
providers to build a DAG. When a non-source batch provider is asked for a
batch, it passes the request on *upstream* (i.e., towards a source) to receive
a batch, possibly modifies it, and passes the batch on *downstream*.

As an example, this scheme allows the modelling of on-the-fly data augmentation
as a batch provider:

.. code-block:: python

  augment =
      gp.ElasticAugment(
          control_point_spacing=[4, 40, 40],
          jitter_sigma=[0, 2, 2],
          rotation_interval=[0, math.pi/2.0])

When :class:`ElasticAugment` is asked for a batch via a request, the
request is automatically changed to request an upstream batch large enough to
perform the elastic augmentation seamlessly.

Another example is the random selection of locations inside a source:

.. code-block:: python

  random =
      gp.RandomLocation()

:class:`RandomLocation` does only modify the request (by changing the offset).

Training
--------

:class:`Training<tensorflow.Train>` itself is modelled as a batch provider. It
takes a batch and performs one training iteration:

.. code-block:: python

  train =
      gp.tensorflow.Train(...)

Putting it together, a very simple pipeline for training 1000 iterations would
be

.. code-block:: python

  pipeline = source + random + augment + train

  request = gp.BatchRequest()
  request.add(raw, (84, 268, 268))
  request.add(gt, (56, 56, 56))
  request.add(gt_mask, (56, 56, 56))

  with gp.build(pipeline) as p:
      for i in range(1000):
      p.request_batch(request)

Note that we use a :class:`BatchRequest` object to communicate
downstream the requirements for a batch. In the example, we are interested in
batches of certain sizes (fitting the network we want to train) with raw data,
ground-truth labels, and a mask.

Going Further
-------------

``gunpowder`` provides much more nodes to chain together, including a
:ref:`pre-cache<sec_api_precache>` node for easy parallel fetching of batches,
several :ref:`augmentation nodes<sec_api_augmentation>`, and nodes for
:ref:`profiling<sec_api_profiling>` and :ref:`inspection<sec_api_snapshot>`.
For a complete list see the :ref:`API reference<sec_api>`.

Continue reading :ref:`here<sec_custom_providers>` to learn how to write your
own ``gunpowder`` batch providers.

Working examples (with many more batch providers) can be found in `the example
directory <https://github.com/funkey/gunpowder/tree/master/examples>`_.
