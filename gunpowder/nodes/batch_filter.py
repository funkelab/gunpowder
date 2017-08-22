import copy
import logging

from .batch_provider import BatchProvider
from gunpowder.profiling import Timing

logger = logging.getLogger(__name__)

class BatchFilter(BatchProvider):
    '''Convenience wrapper for BatchProviders with exactly one input provider.

    By default, a node of this class will expose the same :class:`ProviderSpec` 
    as the upstream provider. You can modify the provider spec by calling 
    :fun:`add_volume_spec`, :fun:`add_points_spec`, :fun:`update_volume_spec`, 
    and :fun:`update_points_spec` in :fun:`setup`.

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
            'process'. Use it to modify a batch request to be passed 
            upstream.
    '''

    def get_upstream_provider(self):
        assert len(self.get_upstream_providers()) == 1, "BatchFilters need to have exactly one upstream provider"
        return self.get_upstream_providers()[0]

    def add_volume_spec(self, volume_type, volume_spec):
        '''Introduce a new volume spec provided by this `BatchFilter`. Asserts 
        that no upstream volume spec for the same volume type is shadowed.
        '''
        assert volume_type not in self.get_spec().volume_specs, "Node %s is trying to add spec for %s, but is already provided upstream."%(type(self).__name__, volume_type)
        super(BatchFilter, self).add_volume_spec(volume_type, volume_spec)

    def add_points_spec(self, points_type, points_spec):
        '''Introduce a new points spec provided by this `BatchFilter`. Asserts 
        that no upstream points spec for the same volume type is shadowed.
        '''
        assert points_type not in self.get_spec().points_specs, "Node %s is trying to add spec for %s, but is already provided upstream."%(type(self).__name__, points_type)
        super(BatchFilter, self).add_points_spec(points_type, points_spec)

    def update_volume_spec(self, volume_type, volume_spec):
        '''Change the spec of a volume provided by this `BatchFilter`. Asserts 
        that the volume type is provided upstream.
        '''
        assert volume_type in self.get_spec().volume_specs, "Node %s is trying to change the spec for %s, but is not provided upstream."%(type(self).__name__, volume_type)
        self.get_spec().volume_specs[volume_type] = copy.deepcopy(volume_spec)
        logger.debug("%s updates %s with %s"%(self.name(), volume_type, volume_spec))

    def update_points_spec(self, points_type, points_spec):
        '''Change the spec of points provided by this `BatchFilter`. Asserts 
        that the points type is provided upstream.
        '''
        assert points_type in self.get_spec().points_specs, "Node %s is trying to change the spec for %s, but is not provided upstream."%(type(self).__name__, points_type)
        self.get_spec().points_specs[points_type] = copy.deepcopy(points_spec)
        logger.debug("%s updates %s with %s"%(self.name(), points_type, points_spec))

    def get_spec(self):

        if not hasattr(self, 'spec'):
            self.spec = copy.deepcopy(self.get_upstream_provider().get_spec())
            assert self.spec is not None, "No provider spec was set and the upstream provider spec is not defined. Make sure you ask for the spec after 'build'ing your graph, or in 'setup' of your node."

        return self.spec

    def provide(self, request):

        # operate on a copy of the request, to provide the original request to 
        # 'process' for convenience
        upstream_request = copy.deepcopy(request)

        timing_prepare = Timing(self, 'prepare')

        timing_prepare.start()
        self.prepare(upstream_request)
        timing_prepare.stop()

        batch = self.get_upstream_provider().request_batch(upstream_request)

        timing_process = Timing(self, 'process')

        timing_process.start()
        self.process(batch, request)
        timing_process.stop()

        batch.profiling_stats.add(timing_prepare)
        batch.profiling_stats.add(timing_process)

        return batch

    def prepare(self, request):
        '''To be implemented in subclasses.

        Prepare for a batch request. Change the request as needed, it will be 
        passed on upstream.
        '''
        pass

    def process(self, batch, request):
        '''To be implemented in subclasses.

        Filter a batch, will be called after 'prepare'. Change batch as needed, 
        it will be passed downstream. 'request' is the same as passed to 
        'prepare', provided for convenience.
        '''
        raise RuntimeError("Class %s does not implement 'process'"%type(self).__name__)
