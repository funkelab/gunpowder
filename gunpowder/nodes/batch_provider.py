import copy
import logging
from gunpowder.coordinate import Coordinate
from gunpowder.provider_spec import ProviderSpec

logger = logging.getLogger(__name__)

class BatchProvider(object):
    '''Superclass for all nodes in a `gunpowder` graph.

    A `BatchProvider` provides :class:`Batch`es containing :class:`Volume`s 
    and/or :class:`Points`. The available types and ROIs `Volume`s and `Points` 
    are specified in a :class:`ProviderSpec` instance, accessible via 
    `get_spec`.

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

        In setup, call :fun:`add_volume_spec` and :fun:`add_points_spec` to 
        announce the volumes and points provided by this node.
        '''
        pass

    def teardown(self):
        '''To be implemented in subclasses.

        Called during destruction of the DAG. Subclasses should use this to stop 
        worker processes, if they used some.
        '''
        pass

    def add_volume_spec(self, volume_type, volume_spec):
        '''Introduce a new volume spec provided by this `BatchFilter`.
        '''
        self.get_spec().volume_specs[volume_type] = copy.deepcopy(volume_spec)
        logger.debug("%s provides new %s with %s"%(self.name(), volume_type, volume_spec))

    def add_points_spec(self, points_type, points_spec):
        '''Introduce a new points spec provided by this `BatchFilter`.
        '''
        self.get_spec().points_specs[points_type] = copy.deepcopy(points_spec)
        logger.debug("%s provides new %s with %s"%(self.name(), points_type, points_spec))

    def get_spec(self):
        '''Get the :class:`ProviderSpec` of this `BatchProvider`.
        '''
        if not hasattr(self, 'spec'):
            self.spec = ProviderSpec()
        return self.spec

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

        spec = self.get_spec()

        for (volume_type, request_spec) in request.volume_specs.items():

            assert volume_type in spec.volume_specs, "%s: Asked for %s which this node does not provide"%(self.name(), volume_type)

            provided_spec = spec.volume_specs[volume_type]

            provided_roi = provided_spec.roi
            request_roi = request_spec.roi

            if provided_roi is not None:
                assert provided_roi.contains(request_roi), "%s: %s's ROI %s outside of my ROI %s"%(self.name(), volume_type, request_roi, provided_roi)

            if request_spec.voxel_size is not None:
                assert provided_spec.voxel_size == request_spec.voxel_size, "%s: voxel size %s requested for %s, but this node provides %s"%(
                        request_spec.voxel_size,
                        volume_type,
                        provided_spec.voxel_size)

            for d in range(request_roi.dims()):
                assert request_roi.get_shape()[d]%provided_spec.voxel_size[d] == 0, \
                        "in request %s, dimension %d of request %s is not a multiple of voxel_size %d"%(
                                request,
                                d,
                                volume_type,
                                provided_spec.voxel_size[d])

        for (points_type, request_spec) in request.points_specs.items():

            assert points_type in spec.points_specs, "%s: Asked for %s which this node does not provide"%(self.name(), points_type)

            provided_roi = spec.points_specs[points_type].roi
            request_roi = request_spec.roi

            if provided_roi is not None:
                assert provided_roi.contains(request_roi), "%s: %s's ROI %s outside of my ROI %s"%(self.name(), points_type, request_roi, provided_roi)

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
            # ensure that the spatial dimensions are the same (other dimensions 
            # on top are okay, e.g., for affinities)
            dims = request_spec.roi.dims()
            data_shape = Coordinate(volume.data.shape[-dims:])
            voxel_size = self.get_spec().volume_specs[volume_type].voxel_size
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

        return self.name() + ": " + str(self.get_spec())
