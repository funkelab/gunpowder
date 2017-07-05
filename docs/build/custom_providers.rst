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
      pass

    def process(self, batch, request):
      pass

``prepare`` and ``process`` will be called in an alternating fashion.
``prepare`` is called first, when a ``BatchRequest`` is passed upstream through
the filter. Your filter is free to change the request in any way it needs to,
for example, by increasing the requested sizes. After ``prepare``, ``process``
will be called with a batch going downstream, which is the upstream's response
to the request you modified in ``prepare``. In ``process``, your filter should
make all necessary changes to the batch and ensure it meets the original
downstream request earlier communicated to ``prepare`` (given as ``request``
parameter in ``process`` for convenience).

For an example of a batch filter changing both the spec going upstream and the
batch going downstream, see :class:`ElasticAugment<gunpowder.ElasticAugment>`.
