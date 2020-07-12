.. _sec_overview:

Overview
========

Many data loading, training, and prediction tasks can be thought of as chaining
a sequence of operations. For example, to train a model to segment cells in a
large 2D+t movie, one might want to:

  1. **Read the data** (the movie) and meta data (spatial resolution, data
     type, ground-truth annotations).
  2. **Pick a sample** to train on from a random frame, at a random location.
  3. **Augment the sample** (rotate, mirror, flip, elastically deform, change
     intensity, etc.).
  4. **Stack several samples** together into a batch.
  5. **Perform a training iteration** on the batch

In ``gunpowder``, a sequence of operations like the example above is assembled
in the form of a **pipeline** of linked **nodes**. Once the pipeline is built,
**requests** can be made at the end of the pipeline (e.g, "give me a batch with
ten images of size ``(100, 100)``"). This request will then be passed upstream
from node to node. Nodes will update the request to ask for additional data
they need (e.g., a node performing a rotation of 45° will require an image of
size ``(142, 142)`` to satisfy the requested size ``(100, 100)``). Once the
request hits a source node, a **batch** with the requested data will be
created. This batch is then sent downstream the pipeline, once again visiting
each node along the path to perform the actual operation (e.g., rotate by 45°
and crop to ``(100, 100)``).

The :ref:`Simple Pipeline<sec_tutorial_simple_pipeline>` demonstrates how such
a pipeline in assembled in ``gunpowder`` (and also how to use a similar
pipeline for prediction).
