.. _sec_nodes:

.. automodule:: gunpowder

Nodes
=====

Nodes in :mod:`gunpowder` take a :class:`BatchRequest` to provide a
:class:`Batch`. They can be chained together to form an acyclic graph. The most
general node is a :class:`BatchProvider`. However, many nodes have only one
upstream provider and can thus be modelled more conveniently as a
:class:`BatchFilter`. See :ref:`sec_custom_providers` for how to write your own node.

  .. autoclass:: BatchProvider
    :members: setup, provides, provide, teardown, spec, request_batch

  .. autoclass:: BatchFilter
    :members: setup, updates, provides, enable_autoskip, prepare, process, teardown, spec, request_batch

  .. autoclass:: ProviderSpec

List of All Gunpowder Nodes
---------------------------

  .. autoclass:: AddGtAffinities

  .. autoclass:: Chunk

  .. autoclass:: DefectAugment

  .. autoclass:: ElasticAugment

  .. autoclass:: ExcludeLabels

  .. autoclass:: GrowBoundary

  .. autoclass:: Hdf5Source

  .. autoclass:: IntensityAugment

  .. autoclass:: IntensityScaleShift

  .. autoclass:: Normalize

  .. autoclass:: Pad

  .. autoclass:: PreCache

  .. autoclass:: PrintProfilingStats

  .. autoclass:: RandomLocation

  .. autoclass:: RandomProvider

  .. autoclass:: Reject

  .. autoclass:: SimpleAugment

  .. autoclass:: Snapshot

  .. autoclass:: SplitAndRenumberSegmentationLabels

  .. autoclass:: ZeroOutConstSections
