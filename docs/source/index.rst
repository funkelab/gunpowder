.. gunpowder documentation master file, created by
   sphinx-quickstart on Fri Jun 30 12:59:21 2017.

Gunpowder Documentation
=======================

What is Gunpowder?
^^^^^^^^^^^^^^^^^^

Gunpowder is a library to facilitate machine learning on large,
multi-dimensional arrays.

Gunpowder allows you to assemble a pipeline from :ref:`data
loading <sec_api_source_nodes>` over
:ref:`pre-processing <sec_api_image_processing_nodes>`,
:ref:`random batch
sampling <sec_api_random_location>`, :ref:`data
augmentation <sec_api_augmentation_nodes>`,
:ref:`pre-caching <sec_api_performance_nodes>`,
:ref:`training/prediction <sec_api_training_and_prediction_nodes>`,
to :ref:`storage of results <sec_api_output_nodes>`
on arbitrarily large volumes of multi-dimensional images. Gunpowder is not
tied to a particular learning framework, and thus complements libraries like
`PyTorch <https://pytorch.org/>`_ or `TensorFlow
<https://www.tensorflow.org/>`_.

.. toctree::
  :maxdepth: 2

  overview
  install
  tutorial_simple_pipeline
  tutorial_batch_provider
  api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
