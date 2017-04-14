from batch_provider import BatchProvider

class BatchProviderChain(BatchProvider):

    def __init__(self, input_provider=None, output_provider=None):
        self.input = input_provider
        self.output = output_provider

    def initialize_all(self):
        return self.output.initialize_all()

    def add_upstream_provider(self, batch_provider):
        return self.input.add_upstream_provider(batch_provider)

    def get_upstream_providers(self):
        return self.input.get_upstream_providers()

    def get_spec(self):
        return self.output.get_spec()

    def request_batch(self, batch_spec):
        return self.output.request_batch(batch_spec)

    def __add__(self, batch_provider):

        assert isinstance(batch_provider, BatchProvider), "Can only add BatchProvider to provider chains."
        print("BatchProviderChain: linking %s -> %s"%(str(self.output),str(batch_provider)))

        batch_provider.add_upstream_provider(self.output)
        return BatchProviderChain(self.input, batch_provider)

def batch_provider_add(self, batch_provider):

    assert isinstance(batch_provider, BatchProvider), "Can only add BatchProvider to batch providers."
    print("BatchProviderChain: linking %s -> %s"%(str(self),str(batch_provider)))

    batch_provider.add_upstream_provider(self)
    return BatchProviderChain(self, batch_provider)

BatchProvider.__add__ = batch_provider_add
