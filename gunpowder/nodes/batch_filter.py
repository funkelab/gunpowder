import copy
import logging

from .batch_provider import BatchProvider
from gunpowder.profiling import Timing

logger = logging.getLogger(__name__)

class BatchFilter(BatchProvider):
    '''Convenience wrapper for :class:`BatchProviders<BatchProvider>` with
    exactly one input provider.

    By default, a node of this class will expose the same :class:`ProviderSpec`
    as the upstream provider. You can modify the provider spec by calling
    :func:`provides` and :func:`updates` in :func:`setup`.

    Subclasses need to implement at least :func:`process` to modify a passed
    batch (downstream). Optionally, the following methods can be implemented:

        :func:`setup`

            Initialize this filter. Called after setup of the DAG. All upstream 
            providers will be set up already.

        :func:`teardown`

            Destruct this filter, free resources, stop worker processes.

        :func:`prepare`

            Prepare for a batch request. Always called before each 
            :func:`process`. Use it to modify a batch request to be passed
            upstream.
    '''

    def get_upstream_provider(self):
        assert len(self.get_upstream_providers()) == 1, "BatchFilters need to have exactly one upstream provider"
        return self.get_upstream_providers()[0]

    def updates(self, key, spec):
        '''Update an output provided by this :class:`BatchFilter`.

        Implementations should call this in their :func:`setup` method, which
        will be called when the pipeline is build.

        Args:

            key (:class:`ArrayKey` or :class:`PointsKey`):

                The array or point set key this filter updates.

            spec (:class:`ArraySpec` or :class:`PointsSpec`):

                The updated spec of the array or point set.
        '''

        assert key in self.spec, "Node %s is trying to change the spec for %s, but is not provided upstream."%(type(self).__name__, key)
        self.spec[key] = copy.deepcopy(spec)
        self.updated_items.append(key)

        logger.debug("%s updates %s with %s"%(self.name(), key, spec))

    def enable_autoskip(self, skip=True):
        '''Enable automatic skipping of this :class:`BatchFilter`, based on
        given :func:`updates` and :func:`provides` calls. Has to be called in
        :func:`setup`.

        By default, :class:`BatchFilters<BatchFilter>` are not skipped
        automatically, regardless of what they update or provide. If autskip is
        enabled, :class:`BatchFilters<BatchFilter>` will only be run if the
        request contains at least one key reported earlier with
        :func:`updates` or :func:`provides`.
        '''

        self._autoskip_enabled = skip

    def _init_spec(self):
        # default for BatchFilters is to provide the same as upstream
        if not hasattr(self, '_spec') or self._spec is None:
            self._spec = copy.deepcopy(self.get_upstream_provider().spec)

    def internal_teardown(self):

        logger.debug("Resetting spec of %s", self.name())
        self._spec = None
        self._updated_items = []

        self.teardown()

    @property
    def updated_items(self):
        '''Get a list of the keys that are updated by this `BatchFilter`.

        This list is only available after the pipeline has been build. Before
        that, it is empty.
        '''

        if not hasattr(self, '_updated_items'):
            self._updated_items = []

        return self._updated_items

    @property
    def autoskip_enabled(self):

        if not hasattr(self, '_autoskip_enabled'):
            self._autoskip_enabled = False

        return self._autoskip_enabled

    def provide(self, request):

        # operate on a copy of the request, to provide the original request to
        # 'process' for convenience
        upstream_request = copy.deepcopy(request)

        skip = self.__can_skip(request)

        timing_prepare = Timing(self, 'prepare')
        timing_prepare.start()

        if not skip:
            self.prepare(upstream_request)
            self.remove_provided(upstream_request)

        timing_prepare.stop()

        batch = self.get_upstream_provider().request_batch(upstream_request)

        timing_process = Timing(self, 'process')
        timing_process.start()

        if not skip:
            self.process(batch, request)

        timing_process.stop()

        batch.profiling_stats.add(timing_prepare)
        batch.profiling_stats.add(timing_process)

        return batch

    def __can_skip(self, request):
        '''Check if this filter needs to be run for the given request.'''

        if not self.autoskip_enabled:
            return False

        for key, _ in request.items():
            if key in self.provided_items:
                return False
            if key in self.updated_items:
                return False

        return True

    def setup(self):
        '''To be implemented in subclasses.

        Called during initialization of the DAG. Callees can assume that all
        upstream providers are set up already.

        In setup, call :func:`provides` or :func:`updates` to announce the
        arrays and points provided or changed by this node.
        '''
        pass

    def prepare(self, request):
        '''To be implemented in subclasses.

        Prepare for a batch request. Change the request as needed, it will be
        passed on upstream.
        '''
        pass

    def process(self, batch, request):
        '''To be implemented in subclasses.

        Filter a batch, will be called after :func:`prepare`. Change batch as
        needed, it will be passed downstream. ``request`` is the same as passed
        to :func:`prepare`, provided for convenience.

        Args:

            batch (:class:`Batch`):

                The batch received from upstream to be modified by this node.

            request (:class:`BatchRequest`):

                The request this node received. The updated batch should meet
                this request.
        '''
        raise RuntimeError("Class %s does not implement 'process'"%type(self).__name__)
