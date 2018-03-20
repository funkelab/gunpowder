import logging
import traceback

from gunpowder.nodes.batch_provider import BatchProvider

logger = logging.getLogger(__name__)

class BatchProviderTree(BatchProvider):

    def __init__(self, inputs=None, output=None):
        self.inputs = inputs
        self.output = output
        self.initialized = False

    def setup(self):
        if not self.initialized:
            self.__rec_setup(self.output)
            self.initialized = True
        else:
            logger.warning("batch provider setup() called more than once")

    def internal_teardown(self):
        self.__rec_teardown(self.output)
        self.initialized = False

    def add_upstream_provider(self, batch_provider):
        for input in self.inputs:
            input.add_upstream_provider(batch_provider)

    def get_upstream_providers(self):
        upstream_providers = []
        for input in self.inputs:
            upstream_providers += input.get_upstream_providers()
        return upstream_providers

    @property
    def spec(self):
        return self.output.spec

    def provide(self, request):

        if not self.initialized:
            raise RuntimeError("You are requesting a batch from an uninitialized provider ('setup()' has not been called). Avoid this by using the 'gunpowder.build' context manager, which also takes care of tearing the provider down if it is not used anymore.")

        return self.output.request_batch(request)

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

    def __rec_setup(self, provider):

        for upstream_provider in provider.get_upstream_providers():
            self.__rec_setup(upstream_provider)
        provider.setup()

    def __rec_teardown(self, provider):

        for upstream_provider in provider.get_upstream_providers():
            self.__rec_teardown(upstream_provider)

        try:
            provider.internal_teardown()
        except Exception as e:
            # don't stop on exceptions, try to tear down as much as possible of
            # the DAG
            logger.error("encountered exception during internal teardown: " + str(e))
            traceback.print_exc()

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
