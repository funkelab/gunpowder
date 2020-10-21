import copy
import logging
from gunpowder.coordinate import Coordinate
from gunpowder.points_spec import PointsSpec
from gunpowder.provider_spec import ProviderSpec
from gunpowder.array import ArrayKey
from gunpowder.array_spec import ArraySpec
from gunpowder.graph import GraphKey
from gunpowder.graph_spec import GraphSpec

logger = logging.getLogger(__name__)


class BatchRequestError(Exception):

    def __init__(self, provider, request, batch):
        self.provider = provider
        self.request = request
        self.batch = batch

    def __str__(self):

        return \
            f"Exception in {self.provider.name()} while processing request" \
            f"{self.request} \n" \
            "Batch returned so far:\n" \
            f"{self.batch}"


class BatchProvider(object):
    '''Superclass for all nodes in a `gunpowder` graph.

    A :class:`BatchProvider` provides :class:`Batches<Batch>` containing
    :class:`Arrays<Array>` and/or :class:`Graph`. The available data is
    specified in a :class:`ProviderSpec` instance, accessible via :attr:`spec`.

    To create a new node, subclass this class and implement (at least)
    :func:`setup` and :func:`provide`.

    A :class:`BatchProvider` can be linked to any number of other
    :class:`BatchProviders<BatchProvider>` upstream. If your node accepts
    exactly one upstream provider, consider subclassing :class:`BatchFilter`
    instead.
    '''

    def add_upstream_provider(self, provider):
        self.get_upstream_providers().append(provider)
        return provider

    def remove_upstream_providers(self):
        self.upstream_providers = []

    def get_upstream_providers(self):
        if not hasattr(self, 'upstream_providers'):
            self.upstream_providers = []
        return self.upstream_providers

    def setup(self):
        '''To be implemented in subclasses.

        Called during initialization of the DAG. Callees can assume that all
        upstream providers are set up already.

        In setup, call :func:`provides` to announce the arrays and points
        provided by this node.
        '''
        raise NotImplementedError("Class %s does not implement 'setup'"%self.name())

    def teardown(self):
        '''To be implemented in subclasses.

        Called during destruction of the DAG. Subclasses should use this to
        stop worker processes, if they used some.
        '''
        pass

    def provides(self, key, spec):
        '''Introduce a new output provided by this :class:`BatchProvider`.

        Implementations should call this in their :func:`setup` method, which
        will be called when the pipeline is build.

        Args:

            key (:class:`ArrayKey` or :class:`GraphKey`):

                The array or point set key provided.

            spec (:class:`ArraySpec` or :class:`GraphSpec`):

                The spec of the array or point set provided.
        '''

        logger.debug("Current spec of %s:\n%s", self.name(), self.spec)

        if self.spec is None:
            self._spec = ProviderSpec()

        assert key not in self.spec, (
            "Node %s is trying to add spec for %s, but is already "
            "provided."%(type(self).__name__, key))

        self.spec[key] = copy.deepcopy(spec)
        self.provided_items.append(key)

        logger.debug("%s provides %s with spec %s", self.name(), key, spec)

    def _init_spec(self):
        if not hasattr(self, '_spec'):
            self._spec = None

    def internal_teardown(self):

        logger.debug("Resetting spec of %s", self.name())
        self._spec = None
        self._provided_items = []

        self.teardown()

    @property
    def spec(self):
        '''Get the :class:`ProviderSpec` of this :class:`BatchProvider`.

        Note that the spec is only available after the pipeline has been build.
        Before that, it is ``None``.
        '''
        self._init_spec()
        return self._spec

    @property
    def provided_items(self):
        '''Get a list of the keys provided by this :class:`BatchProvider`.

        This list is only available after the pipeline has been build. Before
        that, it is empty.
        '''

        if not hasattr(self, '_provided_items'):
            self._provided_items = []

        return self._provided_items

    def remove_provided(self, request):
        '''Remove keys from `request` that are provided by this
        :class:`BatchProvider`.
        '''

        for key in self.provided_items:
            if key in request:
                del request[key]

    def request_batch(self, request):
        '''Request a batch from this provider.

        Args:

            request (:class:`BatchRequest`):

                A request containing (possibly partial)
                :class:`ArraySpecs<ArraySpec>` and
                :class:`GraphSpecs<GraphSpec>`.
        '''

        batch = None

        try:

            logger.debug("%s got request %s", self.name(), request)

            self.check_request_consistency(request)

            batch = self.provide(request.copy())

            self.check_batch_consistency(batch, request)

            self.remove_unneeded(batch, request)

            logger.debug("%s provides %s", self.name(), batch)

        except Exception as e:

            raise BatchRequestError(self, request, batch) from e

        return batch

    def check_request_consistency(self, request):

        for (key, request_spec) in request.items():

            assert key in self.spec, "%s: Asked for %s which this node does not provide"%(self.name(), key)
            assert (
                isinstance(request_spec, ArraySpec) or
                isinstance(request_spec, PointsSpec) or
                isinstance(request_spec, GraphSpec)), ("spec for %s is of type"
                                                        "%s"%(
                                                            key,
                                                            type(request_spec)))

            provided_spec = self.spec[key]

            provided_roi = provided_spec.roi
            request_roi = request_spec.roi

            if provided_roi is not None:
                assert provided_roi.contains(request_roi), "%s: %s's ROI %s outside of my ROI %s"%(self.name(), key, request_roi, provided_roi)

            if isinstance(key, ArrayKey):

                if request_spec.voxel_size is not None:
                    assert provided_spec.voxel_size == request_spec.voxel_size, "%s: voxel size %s requested for %s, but this node provides %s"%(
                            self.name(),
                            request_spec.voxel_size,
                            key,
                            provided_spec.voxel_size)

                if (
                        request_roi is not None and
                        provided_spec.voxel_size is not None):

                    for d in range(request_roi.dims()):
                        assert request_roi.get_shape()[d]%provided_spec.voxel_size[d] == 0, \
                                "in request %s, dimension %d of request %s is not a multiple of voxel_size %d"%(
                                        request,
                                        d,
                                        key,
                                        provided_spec.voxel_size[d])

            if isinstance(key, GraphKey):

                if request_spec.directed is not None:
                    assert request_spec.directed == provided_spec.directed, (
                        f"asked for {key}:  directed={request_spec.directed} but "
                        f"{self.name()} provides directed={provided_spec.directed}"
                    )
    def check_batch_consistency(self, batch, request):

        for (array_key, request_spec) in request.array_specs.items():

            assert array_key in batch.arrays, "%s requested, but %s did not provide it."%(array_key,self.name())
            array = batch.arrays[array_key]
            assert array.spec.roi == request_spec.roi, "%s ROI %s requested, but ROI %s provided by %s."%(
                    array_key,
                    request_spec.roi,
                    array.spec.roi,
                    self.name()
            )
            assert array.spec.voxel_size == self.spec[array_key].voxel_size, (
                "voxel size of %s announced, but %s "
                "delivered for %s"%(
                    self.spec[array_key].voxel_size,
                    array.spec.voxel_size,
                    array_key))
            # ensure that the spatial dimensions are the same (other dimensions 
            # on top are okay, e.g., for affinities)
            if request_spec.roi is not None:
                dims = request_spec.roi.dims()
                data_shape = Coordinate(array.data.shape[-dims:])
                voxel_size = self.spec[array_key].voxel_size
                assert data_shape == request_spec.roi.get_shape()/voxel_size, "%s ROI %s requested, but size of array is %s*%s=%s provided by %s."%(
                        array_key,
                        request_spec.roi,
                        data_shape,
                        voxel_size,
                        data_shape*voxel_size,
                        self.name()
                )
            if request_spec.dtype is not None:
                assert batch[array_key].data.dtype == request_spec.dtype, \
                    "dtype of array %s (%s) does not match requested dtype %s" % (
                        array_key,
                        batch[array_key].data.dtype,
                        request_spec.dtype)

        for (graph_key, request_spec) in request.graph_specs.items():

            assert graph_key in batch.graphs, "%s requested, but %s did not provide it."%(graph_key,self.name())
            graph = batch.graphs[graph_key]
            assert graph.spec.roi == request_spec.roi, "%s ROI %s requested, but ROI %s provided by %s."%(
                                            graph_key,
                                            request_spec.roi,
                                            graph.spec.roi,
                                            self.name())

            if request_spec.directed is not None:
                assert request_spec.directed == graph.directed, (
                    f"Recieved {graph_key}:  directed={graph.directed} but "
                    f"{self.name()} should provide directed={request_spec.directed}"
                )

            for node in graph.nodes:
                contained = graph.spec.roi.contains(node.location)
                dangling = not contained and all(
                    [
                        graph.spec.roi.contains(v.location)
                        for v in graph.neighbors(node)
                    ]
                )
                assert contained or dangling, (
                    f"graph {graph_key} provided by {self.name()} with ROI {graph.spec.roi} "
                    f"contain point at {node.location} which is neither contained nor "
                    f"'dangling'"
                )

    def remove_unneeded(self, batch, request):

        batch_keys = set(list(batch.arrays.keys()) + list(batch.graphs.keys()))
        for key in batch_keys:
            if key not in request:
                del batch[key]

    def provide(self, request):
        '''To be implemented in subclasses.

        This function takes a :class:`BatchRequest` and should return the
        corresponding :class:`Batch`.

        Args:

            request(:class:`BatchRequest`):

                The request to process.
        '''
        raise NotImplementedError("Class %s does not implement 'provide'"%self.name())

    def name(self):
        return type(self).__name__

    def __repr__(self):

        return self.name() + ", providing: " + str(self.spec)
