.. _sec_api:

API Reference
=============

.. automodule:: gunpowder
   :no-index:

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
    :members: items

BatchRequest
^^^^^^^^^^^^
  .. autoclass:: BatchRequest
    :members: add

ArraySpec
^^^^^^^^^
  .. autoclass:: ArraySpec
    :members:

GraphSpec
^^^^^^^^^
  .. autoclass:: GraphSpec
    :members:

Geometry
--------

Coordinate
^^^^^^^^^^
  .. autoclass:: Coordinate
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

ZarrSource
^^^^^^^^^^
  .. autoclass:: ZarrSource

Hdf5Source
^^^^^^^^^^
  .. autoclass:: Hdf5Source

KlbSource
^^^^^^^^^
  .. autoclass:: KlbSource

DvidSource
^^^^^^^^^^
  .. autoclass:: DvidSource

CsvPointsSource
^^^^^^^^^^^^^^^
  .. autoclass:: CsvPointsSource

GraphSource
^^^^^^^^^^^

  .. autoclass:: GraphSource

.. _sec_api_augmentation_nodes:

Augmentation Nodes
------------------

DefectAugment
^^^^^^^^^^^^^
  .. autoclass:: DefectAugment

DeformAugment
^^^^^^^^^^^^^^
  .. autoclass:: DeformAugment

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

SpecifiedLocation
^^^^^^^^^^^^^^^^^
  .. autoclass:: SpecifiedLocation

IterateLocations
^^^^^^^^^^^^^^^^

  .. autoclass:: IterateLocations

.. _sec_api_array_manipulation_nodes:

Array Manipulation Nodes
------------------------

Squeeze
^^^^^^^
  .. autoclass:: Squeeze

Unsqueeze
^^^^^^^^^
  .. autoclass:: Unsqueeze

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

Graph Processing Nodes
----------------------

RasterizeGraph
^^^^^^^^^^^^^^
  .. autoclass:: RasterizeGraph
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


  .. automodule:: gunpowder.jax

jax.Train
^^^^^^^^^^^^^^^^
  .. autoclass:: Train

jax.Predict
^^^^^^^^^^^^^^^^^^
  .. autoclass:: Predict


.. _sec_api_output_nodes:

Output Nodes
------------

.. automodule:: gunpowder
   :no-index:

Hdf5Write
^^^^^^^^^
  .. autoclass:: Hdf5Write

ZarrWrite
^^^^^^^^^
  .. autoclass:: ZarrWrite

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

Iterative Processing Nodes
--------------------------

Scan
^^^^
  .. autoclass:: Scan

DaisyRequestBlocks
^^^^^^^^^^^^^^^^^^
  .. autoclass:: DaisyRequestBlocks
