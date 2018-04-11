.. _sec_api:

API Reference
=============

.. automodule:: gunpowder

Data Containers
---------------

Batch
^^^^^
  .. autoclass:: Batch
    :members: arrays, points, items

Array
^^^^^
  .. autoclass:: Array
    :members:

Points
^^^^^^
  .. autoclass:: Points
    :members:

Point
^^^^^
  .. autoclass:: Point
    :members:

ArrayKey
^^^^^^^^
  .. autoclass:: ArrayKey

PointsKey
^^^^^^^^^
  .. autoclass:: PointsKey

Requests and Specifications
---------------------------

ProviderSpec
^^^^^^^^^^^^
  .. autoclass:: ProviderSpec
    :members: array_specs, points_specs, items

BatchRequest
^^^^^^^^^^^^
  .. autoclass:: BatchRequest
    :members: add

ArraySpec
^^^^^^^^^
  .. autoclass:: ArraySpec
    :members:

PointsSpec
^^^^^^^^^^
  .. autoclass:: PointsSpec
    :members:

Geometry
--------

Coordinate
^^^^^^^^^^
  .. autoclass:: Coordinate
    :members:

Roi
^^^
  .. autoclass:: Roi
    :members:

Node Base Classes
-----------------

BatchProvider
^^^^^^^^^^^^^
  .. autoclass:: BatchProvider
    :members: setup, provides, provide, teardown, spec, request_batch

BatchFilter
^^^^^^^^^^^
  .. autoclass:: BatchFilter
    :members: setup, updates, provides, enable_autoskip, prepare, process, teardown, spec, request_batch

Source Nodes
------------

CsvPointsSource
^^^^^^^^^^^^^^^
  .. autoclass:: CsvPointsSource

DvidSource
^^^^^^^^^^
  .. autoclass:: DvidSource

Hdf5Source
^^^^^^^^^^
  .. autoclass:: Hdf5Source

ZarrSource
^^^^^^^^^^
  .. autoclass:: ZarrSource

N5Source
^^^^^^^^
  .. autoclass:: N5Source

KlbSource
^^^^^^^^^
  .. autoclass:: KlbSource

.. _sec_api_augmentation:

Augmentation Nodes
------------------

DefectAugment
^^^^^^^^^^^^^
  .. autoclass:: DefectAugment

ElasticAugment
^^^^^^^^^^^^^^
  .. autoclass:: ElasticAugment

IntensityAugment
^^^^^^^^^^^^^^^^
  .. autoclass:: IntensityAugment

SimpleAugment
^^^^^^^^^^^^^
  .. autoclass:: SimpleAugment

Location Manipulation Nodes
---------------------------

Crop
^^^^
  .. autoclass:: Crop

Pad
^^^
  .. autoclass:: Pad

RandomLocation
^^^^^^^^^^^^^^
  .. autoclass:: RandomLocation

Reject
^^^^^^
  .. autoclass:: Reject

Scan
^^^^
  .. autoclass:: Scan

SpecifiedLocation
^^^^^^^^^^^^^^^^^
  .. autoclass:: SpecifiedLocation

Image Processing Nodes
----------------------

DownSample
^^^^^^^^^^
  .. autoclass:: DownSample

IntensityScaleShift
^^^^^^^^^^^^^^^^^^^
  .. autoclass:: IntensityScaleShift

Normalize
^^^^^^^^^
  .. autoclass:: Normalize

Label Manipulation Nodes
------------------------

AddAffinities
^^^^^^^^^^^^^
  .. autoclass:: AddAffinities

BalanceLabels
^^^^^^^^^^^^^
  .. autoclass:: BalanceLabels

ExcludeLabels
^^^^^^^^^^^^^
  .. autoclass:: ExcludeLabels

GrowBoundary
^^^^^^^^^^^^
  .. autoclass:: GrowBoundary

RenumberConnectedComponents
^^^^^^^^^^^^^^^^^^^^^^^^^^^
  .. autoclass:: RenumberConnectedComponents

Point Processing Nodes
----------------------

RasterizePoints
^^^^^^^^^^^^^^^
  .. autoclass:: RasterizePoints
  .. autoclass:: RasterizationSettings

Provider Combination Nodes
--------------------------

MergeProvider
^^^^^^^^^^^^^
  .. autoclass:: MergeProvider

RandomProvider
^^^^^^^^^^^^^^
  .. autoclass:: RandomProvider

Training and Prediction Nodes
-----------------------------
  .. automodule:: gunpowder.caffe

caffe.Train
^^^^^^^^^^^
  .. autoclass:: Train
  .. autoclass:: SolverParameters

caffe.Predict
^^^^^^^^^^^^^
  .. autoclass:: Predict

  .. automodule:: gunpowder.tensorflow

tensorflow.Train
^^^^^^^^^^^^^^^^
  .. autoclass:: Train

tensorflow.Predict
^^^^^^^^^^^^^^^^^^
  .. autoclass:: Predict

  .. automodule:: gunpowder

Output Nodes
------------

Hdf5Write
^^^^^^^^^
  .. autoclass:: Hdf5Write

.. _sec_api_snapshot:

Snapshot
^^^^^^^^
  .. autoclass:: Snapshot

Performance Nodes
-----------------

.. _sec_api_precache:

PreCache
^^^^^^^^
  .. autoclass:: PreCache

.. _sec_api_profiling:

PrintProfilingStats
^^^^^^^^^^^^^^^^^^^
  .. autoclass:: PrintProfilingStats
