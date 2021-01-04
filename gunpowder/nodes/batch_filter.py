import copy
import logging

from .batch_provider import BatchProvider
from gunpowder.batch_request import BatchRequest
from gunpowder.profiling import Timing

logger = logging.getLogger(__name__)


class BatchFilterError(Exception):

    def __init__(self, batch_filter, msg):
        self.batch_filter = batch_filter
        self.msg = msg

    def __str__(self):

        return f"Error in {self.batch_filter.name()}: {self.msg}"


class BatchFilter(BatchProvider):
    """Convenience wrapper for :class:`BatchProviders<BatchProvider>` with
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
            :func:`process`. Used to communicate dependencies.
    """

    @property
    def remove_placeholders(self):
        if not hasattr(self, '_remove_placeholders'):
            return False
        return self._remove_placeholders

    def get_upstream_provider(self):
        if len(self.get_upstream_providers()) != 1:
            raise BatchFilterError(
                self,
                "BatchFilters need to have exactly one upstream provider, "
                f"this one has {len(self.get_upstream_providers())}: "
                f"({[b.name() for b in self.get_upstream_providers()]}")
        return self.get_upstream_providers()[0]

    def updates(self, key, spec):
        """Update an output provided by this :class:`BatchFilter`.

        Implementations should call this in their :func:`setup` method, which
        will be called when the pipeline is build.

        Args:

            key (:class:`ArrayKey` or :class:`GraphKey`):

                The array or point set key this filter updates.

            spec (:class:`ArraySpec` or :class:`GraphSpec`):

                The updated spec of the array or point set.
        """

        if key not in self.spec:
            raise BatchFilterError(
                self,
                f"BatchFilter {self} is trying to change the spec for {key}, "
                f"but {key} is not provided upstream. Upstream offers: "
                f"{self.get_upstream_provider().spec}")
        self.spec[key] = copy.deepcopy(spec)
        self.updated_items.append(key)

        logger.debug("%s updates %s with %s" % (self.name(), key, spec))

    def enable_autoskip(self, skip=True):
        """Enable automatic skipping of this :class:`BatchFilter`, based on
        given :func:`updates` and :func:`provides` calls. Has to be called in
        :func:`setup`.

        By default, :class:`BatchFilters<BatchFilter>` are not skipped
        automatically, regardless of what they update or provide. If autskip is
        enabled, :class:`BatchFilters<BatchFilter>` will only be run if the
        request contains at least one key reported earlier with
        :func:`updates` or :func:`provides`.
        """

        self._autoskip_enabled = skip

    def _init_spec(self):
        # default for BatchFilters is to provide the same as upstream
        if not hasattr(self, "_spec") or self._spec is None:
            if len(self.get_upstream_providers()) != 0:
                self._spec = copy.deepcopy(self.get_upstream_provider().spec)
            else:
                self._spec = None

    def internal_teardown(self):

        logger.debug("Resetting spec of %s", self.name())
        self._spec = None
        self._updated_items = []

        self.teardown()

    @property
    def updated_items(self):
        """Get a list of the keys that are updated by this `BatchFilter`.

        This list is only available after the pipeline has been build. Before
        that, it is empty.
        """

        if not hasattr(self, "_updated_items"):
            self._updated_items = []

        return self._updated_items

    @property
    def autoskip_enabled(self):

        if not hasattr(self, "_autoskip_enabled"):
            self._autoskip_enabled = False

        return self._autoskip_enabled

    def provide(self, request):

        skip = self.__can_skip(request)

        timing_prepare = Timing(self, "prepare")
        timing_prepare.start()

        downstream_request = request.copy()

        if not skip:
            dependencies = self.prepare(request)
            if isinstance(dependencies, BatchRequest):
                upstream_request = request.update_with(dependencies)
            elif dependencies is None:
                upstream_request = request.copy()
            else:
                raise BatchFilterError(
                    self,
                    f"This BatchFilter returned a {type(dependencies)}! "
                    "Supported return types are: `BatchRequest` containing your exact "
                    "dependencies or `None`, indicating a dependency on the full request.")
            self.remove_provided(upstream_request)
        else:
            upstream_request = request.copy()
        self.remove_provided(upstream_request)

        timing_prepare.stop()

        batch = self.get_upstream_provider().request_batch(upstream_request)

        timing_process = Timing(self, "process")
        timing_process.start()

        if not skip:
            if dependencies is not None:
                dependencies.remove_placeholders()
                node_batch = batch.crop(dependencies)
            else:
                node_batch = batch
            downstream_request.remove_placeholders()
            processed_batch = self.process(node_batch, downstream_request)
            if processed_batch is None:
                processed_batch = node_batch
            batch = batch.merge(processed_batch, merge_profiling_stats=False).crop(
                downstream_request
            )

        timing_process.stop()

        batch.profiling_stats.add(timing_prepare)
        batch.profiling_stats.add(timing_process)

        return batch

    def __can_skip(self, request):
        """Check if this filter needs to be run for the given request."""

        if not self.autoskip_enabled:
            return False

        for key, spec in request.items():
            if spec.placeholder:
                continue
            if key in self.provided_items:
                return False
            if key in self.updated_items:
                return False

        return True

    def setup(self):
        """To be implemented in subclasses.

        Called during initialization of the DAG. Callees can assume that all
        upstream providers are set up already.

        In setup, call :func:`provides` or :func:`updates` to announce the
        arrays and points provided or changed by this node.
        """
        pass

    def prepare(self, request):
        """To be implemented in subclasses.

        Prepare for a batch request. Should return a :class:`BatchRequest` of
        needed dependencies. If None is returned, it will be assumed that all
        of request is needed.
        """
        return None

    def process(self, batch, request):
        """To be implemented in subclasses.

        Filter a batch, will be called after :func:`prepare`. Should return a
        :class:`Batch` containing modified Arrays and Graphs. Keys in the returned
        batch will replace the associated data in the original batch. If None is
        returned it is assumed that the batch has been modified in place. ``request``
        is the same as passed to :func:`prepare`, provided for convenience.

        Args:

            batch (:class:`Batch`):

                The batch received from upstream to be modified by this node.

            request (:class:`BatchRequest`):

                The request this node received. The updated batch should meet
                this request.
        """
        raise BatchFilterError(
            self,
            "does not implement 'process'")
