from gunpowder.nodes.batch_provider import BatchProvider
from gunpowder.profiling import Timing

class BatchFilter(BatchProvider):
    '''Convenience wrapper for BatchProviders with exactly one input provider.

    Subclasses need to implement at least 'process' to modify a passed batch 
    (downstream). Optionally, the following methods can be implemented:

        setup

            Initialize this filter. Called after setup of the DAG. All upstream 
            providers will be set up already.

        teardown

            Destruct this filter, free resources, stop worker processes.

        get_spec

            Get the spec of this provider. If not implemented, the upstream 
            provider spec is used.

        prepare

            Prepare for a batch request. Always called before each 
            'request_batch'. Use it to modify a batch spec to be passed 
            upstream.
    '''

    def get_upstream_provider(self):
        assert len(self.get_upstream_providers()) == 1, "BatchFilters need to have exactly one upstream provider"
        return self.get_upstream_providers()[0]

    def get_spec(self):
        return self.get_upstream_provider().get_spec()

    def request_batch(self, batch_spec):

        timing = Timing(self)

        timing.start()
        self.prepare(batch_spec)
        timing.stop()

        batch = self.get_upstream_provider().request_batch(batch_spec)

        timing.start()
        self.process(batch)
        timing.stop()

        batch.profiling_stats.add(timing)

        return batch

    def prepare(self, batch_spec):
        '''To be implemented in subclasses.

        Prepare for a batch request. Change the batch_spec as needed, it will be 
        passed on upstream.
        '''
        pass

    def process(self, batch):
        '''To be implemented in subclasses.

        Filter a batch, will be called after 'prepare'. Change batch and its 
        spec as needed, it will be passed downstream.
        '''
        raise RuntimeError("Class %s does not implement 'process'"%self.__class__)
