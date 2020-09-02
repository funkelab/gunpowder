.. _sec_api:

API Reference
=============

.. automodule:: gunpowder
   :noindex:

Data Containers
---------------

Batch
^^^^^
  .. autoclass:: Batch
    :members:

Array
^^^^^
  .. autoclass:: Array
    :members:

Graph
^^^^^
  .. autoclass:: Graph
    :members:

Node
^^^^
  .. autoclass:: Node
    :members:

Edge
^^^^
  .. autoclass:: Edge
    :members:

ArrayKey
^^^^^^^^
  .. autoclass:: ArrayKey

GraphKey
^^^^^^^^
  .. autoclass:: GraphKey

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

.. _sec_api_source_nodes:

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

.. _sec_api_augmentation_nodes:

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

NoiseAugment
^^^^^^^^^^^^^^^^
  .. autoclass:: NoiseAugment

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

.. _sec_api_random_location:

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

.. _sec_api_image_processing_nodes:

Image Processing Nodes
----------------------

DownSample
^^^^^^^^^^
  .. autoclass:: DownSample

UpSample
^^^^^^^^
  .. autoclass:: UpSample

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

.. _sec_api_training_and_prediction_nodes:

Training and Prediction Nodes
-----------------------------

Stack
^^^^^
  .. autoclass:: Stack

  .. automodule:: gunpowder.torch

torch.Train
^^^^^^^^^^^
  .. autoclass:: Train

torch.Predict
^^^^^^^^^^^^^
  .. autoclass:: Predict


  .. automodule:: gunpowder.tensorflow

tensorflow.Train
^^^^^^^^^^^^^^^^
  .. autoclass:: Train

tensorflow.Predict
^^^^^^^^^^^^^^^^^^
  .. autoclass:: Predict


  .. automodule:: gunpowder.caffe

caffe.Train
^^^^^^^^^^^
  .. autoclass:: Train
  .. autoclass:: SolverParameters

caffe.Predict
^^^^^^^^^^^^^
  .. autoclass:: Predict


  .. automodule:: gunpowder
     :noindex:

.. _sec_api_output_nodes:

Output Nodes
------------

Hdf5Write
^^^^^^^^^
  .. autoclass:: Hdf5Write

ZarrWrite
^^^^^^^^^
  .. autoclass:: ZarrWrite

N5Write
^^^^^^^^^
  .. autoclass:: N5Write

.. _sec_api_snapshot:

Snapshot
^^^^^^^^
  .. autoclass:: Snapshot

.. _sec_api_performance_nodes:

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
