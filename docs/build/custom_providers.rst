.. _sec_custom_providers:

Writing Custom Batch Providers
==============================

The simplest batch provider is a :class:`BatchFilter<gunpowder.BatchFilter>`,
which has exactly one upstream provider. To create a new one, subclass it and
override :meth:`prepare<gunpowder.BatchFilter.prepare>` and/or
:meth:`process<gunpowder.BatchFilter.process>`:

.. code-block:: python

  class ExampleFilter(BatchFilter):

    def prepare(self, request):
      # create a new request for this node's dependencies
      dependencies = BatchRequest()
      # [...]
      return dependencies

    def process(self, batch, request):
      # create a new batch for this node's output
      output = Batch()
      # [...]
      return output

``prepare`` and ``process`` will be called in an alternating fashion.
``prepare`` is called first, when a ``BatchRequest`` is passed upstream through
the filter. Your filter has to formulate a new request, stating the
dependencies needed by this filter, e.g., by increasing the requested sizes of
existing arrays in the request.
After ``prepare``, ``process`` will be called with a batch going downstream,
which is the upstream's response to the your request you returned in
``prepare``. In ``process``, your filter has to create a new batch with
expected outputs and ensure it meets the original downstream request earlier
communicated to ``prepare`` (given as ``request`` parameter in ``process`` for
convenience).

For a simple example of a batch filter following this scheme, see
the source code of :class:`DownSample<gunpowder.DownSample>`.
