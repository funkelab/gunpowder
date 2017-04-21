from batch_provider import BatchProvider

import logging
logger = logging.getLogger(__name__)

class BatchProviderTree(BatchProvider):

    def __init__(self, inputs=None, output=None):
        self.inputs = inputs
        self.output = output

    def initialize_all(self):
        return self.output.initialize_all()

    def add_upstream_provider(self, batch_provider):
        for input in self.inputs:
            input.add_upstream_provider(batch_provider)

    def get_upstream_providers(self):
        upstream_providers = []
        for input in self.inputs:
            upstream_providers += input.get_upstream_providers()
        return upstream_providers

    def get_spec(self):
        return self.output.get_spec()

    def request_batch(self, batch_spec):
        return self.output.request_batch(batch_spec)

    def __add__(self, batch_provider):

        assert isinstance(batch_provider, BatchProvider), "Can only add BatchProvider to provider trees."
        logger.debug("linking %s -> %s"%(type(self.output).__name__,type(batch_provider).__name__))

        batch_provider.add_upstream_provider(self.output)
        return BatchProviderTree(self.inputs, batch_provider)

    def __radd__(self, batch_providers):

        assert isinstance(batch_providers, tuple), "Don't know how to r-add anything but tuples."
        for batch_provider in batch_providers:
            assert isinstance(batch_provider, BatchProvider), "Can only add BatchProvider to provider trees."

        for batch_provider in batch_providers:
            logger.debug("linking %s -> %s"%(type(batch_provider).__name__,type(self).__name__))
            self.add_upstream_provider(batch_provider)

        return BatchProviderTree(list(batch_providers), self.output)

def batch_provider_add(self, batch_provider):

    assert isinstance(batch_provider, BatchProvider), "Can only add BatchProvider to batch providers."
    logger.debug("linking %s -> %s"%(type(self).__name__,type(batch_provider).__name__))

    batch_provider.add_upstream_provider(self)
    return BatchProviderTree([self], batch_provider)

def batch_provider_radd(self, batch_providers):

    assert isinstance(batch_providers, tuple), "Don't know how to r-add anything but tuples."
    for batch_provider in batch_providers:
        assert isinstance(batch_provider, BatchProvider), "Can only add BatchProvider to batch providers."

    for batch_provider in batch_providers:
        logger.debug("linking %s -> %s"%(type(batch_provider).__name__,type(self).__name__))
        self.add_upstream_provider(batch_provider)

    return BatchProviderTree(list(batch_providers), self)

BatchProvider.__add__ = batch_provider_add
BatchProvider.__radd__ = batch_provider_radd
