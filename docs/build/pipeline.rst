.. _sec_pipeline:

Pipelines
=========

.. automodule:: gunpowder

A ``gunpowder`` processing pipeline consists of nodes, which can be chained
together to form a directed acyclic graph.

All nodes inherit from :class:`BatchProvider` and can be asked to provide a
batch by formulating a :class:`BatchRequest` and passing it to
:func:`BatchProvider.request_batch`.

See :ref:`sec_custom_providers` for how to write your own node.
