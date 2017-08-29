import copy
import logging
from gunpowder.coordinate import Coordinate
from gunpowder.points_spec import PointsSpec
from gunpowder.provider_spec import ProviderSpec
from gunpowder.volume import VolumeType
from gunpowder.volume_spec import VolumeSpec

logger = logging.getLogger(__name__)

class BatchProvider(object):
    '''Superclass for all nodes in a `gunpowder` graph.

    A `BatchProvider` provides :class:`Batch`es containing :class:`Volume`s 
    and/or :class:`Points`. The available types and ROIs `Volume`s and `Points` 
    are specified in a :class:`ProviderSpec` instance, accessible via 
    `self.spec`.

    To create a new `gunpowder` node, subclass this class and implement (at 
    least) :fun:`setup` and :fun:`provide`.

    A `BatchProvider` can be linked to any number of other `BatchProvider`s 
    upstream. If your node accepts exactly one upstream provider, consider 
    subclassing :class:`BatchFilter` instead.
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

        In setup, call :fun:`provides` to announce the volumes and points 
        provided by this node.
        '''
        raise NotImplementedError("Class %s does not implement 'setup'"%self.name())

    def teardown(self):
        '''To be implemented in subclasses.

        Called during destruction of the DAG. Subclasses should use this to stop 
        worker processes, if they used some.
        '''
        pass

    def provides(self, identifier, spec):
        '''Introduce a new output provided by this `BatchProvider`.

        Implementations should call this in their :fun:`setup` method, which 
        will be called when the pipeline is build.

        Args:

            identifier: A :class:`VolumeType` or `PointsType` instance to refer to the output.

            spec: A :class:`VolumeSpec` or `PointsSpec` to describe the output.
        '''

        if self.spec is None:
            self._spec = ProviderSpec()

        assert identifier not in self.spec, "Node %s is trying to add spec for %s, but is already provided."%(type(self).__name__, identifier)
        self.spec[identifier] = copy.deepcopy(spec)

        logger.debug("%s provides %s with spec %s"%(self.name(), identifier, spec))

    def _init_spec(self):
        if not hasattr(self, '_spec'):
            self._spec = None

    def _reset_spec(self):
        self._spec = None

    @property
    def spec(self):
        '''Get the :class:`ProviderSpec` of this `BatchProvider`.

        Note that the spec is only available after the pipeline has been build. 
        Before that, it is None.
        '''
        self._init_spec()
        return self._spec

    def request_batch(self, request):
        '''Request a batch from this provider.

        Args:

            request(:class:`BatchRequest`): A request containing (possibly 
                partial) :class:`VolumeSpec`s and :class:`PointsSpec`s.
        '''

        logger.debug("%s got request %s"%(self.name(),request))

        self.check_request_consistency(request)

        batch = self.provide(copy.deepcopy(request))

        self.check_batch_consistency(batch, request)

        logger.debug("%s provides %s"%(self.name(),batch))

        return batch

    def check_request_consistency(self, request):

        for (identifier, request_spec) in request.items():

            assert identifier in self.spec, "%s: Asked for %s which this node does not provide"%(self.name(), identifier)
            assert (
                isinstance(request_spec, VolumeSpec) or
                isinstance(request_spec, PointsSpec)), ("spec for %s is of type"
                                                        "%s"%(
                                                            identifier,
                                                            type(request_spec)))

            provided_spec = self.spec[identifier]

            provided_roi = provided_spec.roi
            request_roi = request_spec.roi

            if provided_roi is not None:
                assert provided_roi.contains(request_roi), "%s: %s's ROI %s outside of my ROI %s"%(self.name(), identifier, request_roi, provided_roi)

            if isinstance(identifier, VolumeType):

                if request_spec.voxel_size is not None:
                    assert provided_spec.voxel_size == request_spec.voxel_size, "%s: voxel size %s requested for %s, but this node provides %s"%(
                            request_spec.voxel_size,
                            identifier,
                            provided_spec.voxel_size)

                if request_roi is not None:
                    for d in range(request_roi.dims()):
                        assert request_roi.get_shape()[d]%provided_spec.voxel_size[d] == 0, \
                                "in request %s, dimension %d of request %s is not a multiple of voxel_size %d"%(
                                        request,
                                        d,
                                        identifier,
                                        provided_spec.voxel_size[d])

    def check_batch_consistency(self, batch, request):

        for (volume_type, request_spec) in request.volume_specs.items():

            assert volume_type in batch.volumes, "%s requested, but %s did not provide it."%(volume_type,self.name())
            volume = batch.volumes[volume_type]
            assert volume.spec.roi == request_spec.roi, "%s ROI %s requested, but ROI %s provided by %s."%(
                    volume_type,
                    request_spec.roi,
                    volume.spec.roi,
                    self.name()
            )
            assert volume.spec.voxel_size == self.spec[volume_type].voxel_size, (
                "voxel size of %s announced, but %s "
                "delivered for %s"%(
                    self.spec[volume_type].voxel_size,
                    volume.spec.voxel_size,
                    volume_type))
            # ensure that the spatial dimensions are the same (other dimensions 
            # on top are okay, e.g., for affinities)
            dims = request_spec.roi.dims()
            data_shape = Coordinate(volume.data.shape[-dims:])
            voxel_size = self.spec[volume_type].voxel_size
            assert data_shape == request_spec.roi.get_shape()/voxel_size, "%s ROI %s requested, but size of volume is %s*%s=%s provided by %s."%(
                    volume_type,
                    request_spec.roi,
                    data_shape,
                    voxel_size,
                    data_shape*voxel_size,
                    self.name()
            )

        for (points_type, request_spec) in request.points_specs.items():

            assert points_type in batch.points, "%s requested, but %s did not provide it."%(points_type,self.name())
            points = batch.points[points_type]
            assert points.spec.roi == request_spec.roi, "%s ROI %s requested, but ROI %s provided by %s."%(
                                            points_type,
                                            request_spec.roi,
                                            points.spec.roi,
                                            self.name())

    def provide(self, request):
        '''To be implemented in subclasses.

        Called with a batch request. Should return the requested batch.
        '''
        raise NotImplementedError("Class %s does not implement 'provide'"%self.name())

    def name(self):
        return type(self).__name__

    def __repr__(self):

        return self.name() + ", providing: " + str(self.spec)
