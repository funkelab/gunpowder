import copy
import logging
from gunpowder.coordinate import Coordinate
from gunpowder.points_spec import PointsSpec
from gunpowder.provider_spec import ProviderSpec
from gunpowder.array import ArrayKey
from gunpowder.array_spec import ArraySpec

logger = logging.getLogger(__name__)

class BatchProvider(object):
    '''Superclass for all nodes in a `gunpowder` graph.

    A :class:`BatchProvider` provides :class:`Batches<Batch>` containing
    :class:`Arrays<Array>` and/or :class:`Points`. The available data is
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

            key (:class:`ArrayKey` or :class:`PointsKey`):

                The array or point set key provided.

            spec (:class:`ArraySpec` or :class:`PointsSpec`):

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
                :class:`PointSpecs<PointsSpec>`.
        '''

        logger.debug("%s got request %s", self.name(), request)

        self.check_request_consistency(request)

        batch = self.provide(copy.deepcopy(request))

        self.check_batch_consistency(batch, request)

        logger.debug("%s provides %s", self.name(), batch)

        return batch

    def check_request_consistency(self, request):

        for (key, request_spec) in request.items():

            assert key in self.spec, "%s: Asked for %s which this node does not provide"%(self.name(), key)
            assert (
                isinstance(request_spec, ArraySpec) or
                isinstance(request_spec, PointsSpec)), ("spec for %s is of type"
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

                if request_roi is not None:
                    for d in range(request_roi.dims()):
                        assert request_roi.get_shape()[d]%provided_spec.voxel_size[d] == 0, \
                                "in request %s, dimension %d of request %s is not a multiple of voxel_size %d"%(
                                        request,
                                        d,
                                        key,
                                        provided_spec.voxel_size[d])

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

        for (points_key, request_spec) in request.points_specs.items():

            assert points_key in batch.points, "%s requested, but %s did not provide it."%(points_key,self.name())
            points = batch.points[points_key]
            assert points.spec.roi == request_spec.roi, "%s ROI %s requested, but ROI %s provided by %s."%(
                                            points_key,
                                            request_spec.roi,
                                            points.spec.roi,
                                            self.name())

            for _, point in points.data.items():
                assert points.spec.roi.contains(point.location), (
                    "points provided by %s with ROI %s contain point at %s"%(
                        self.name(), points.spec.roi, point.location))

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
