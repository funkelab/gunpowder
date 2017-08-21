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

    def __init__(self):
        self.spec = ProviderSpec()

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
        self.get_spec().volume_specs[volume_type] = volume_spec

    def add_points_spec(self, points_type, points_spec):
        '''Introduce a new points spec provided by this `BatchFilter`.
        '''
        self.get_spec().points_specs[points_type] = points_spec

    def get_spec(self):
        '''Get the :class:`ProviderSpec` of this `BatchProvider`.
        '''
        return self.spec

    def request_batch(self, request):
        '''Request a batch from this provider.

        Args:

            request(:class:`BatchRequest`): A request containing (possibly 
                partial) :class:`VolumeSpec`s and :class:`PointsSpec`s.
        '''

        logger.debug("%s got request %s"%(type(self).__name__,request))

        upstream_request = copy.deepcopy(request)
        batch = self.provide(upstream_request)

        for (volume_type, spec) in request.volume_specs.items():
            assert volume_type in batch.volumes, "%s requested, but %s did not provide it."%(volume_type,type(self).__name__)
            volume = batch.volumes[volume_type]
            assert volume.spec.roi == spec.roi, "%s ROI %s requested, but ROI %s provided by %s."%(
                    volume_type,
                    spec.roi,
                    volume.spec.roi,
                    type(self).__name__
            )
            # ensure that the spatial dimensions are the same (other dimensions 
            # on top are okay, e.g., for affinities)
            dims = spec.roi.dims()
            data_shape = Coordinate(volume.data.shape[-dims:])
            assert data_shape == spec.roi.get_shape()/spec.voxel_size, "%s ROI %s requested, but size of volume is %s*%s=%s provided by %s."%(
                    volume_type,
                    spec.roi,
                    data_shape,
                    spec.voxel_size,
                    data_shape*spec.voxel_size,
                    type(self).__name__
            )

        for (points_type, spec) in request.points_specs.items():
            assert points_type in batch.points, "%s requested, but %s did not provide it."%(points_type,type(self).__name__)
            points = batch.points[points_type]
            assert points.spec.roi == spec.roi, "%s ROI %s requested, but ROI %s provided by %s."%(
                                            points_type,
                                            spec.roi,
                                            points.spec.roi,
                                            type(self).__name__)

        logger.debug("%s provides %s"%(type(self).__name__,batch))

        return batch

    def provide(self, request):
        '''To be implemented in subclasses.

        Called with a batch request. Should return the requested batch.
        '''
        raise NotImplementedError("Class %s does not implement 'provide'"%type(self).__name__)
