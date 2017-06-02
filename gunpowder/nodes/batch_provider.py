import copy
import logging

logger = logging.getLogger(__name__)

class BatchProvider(object):

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
        '''
        pass

    def teardown(self):
        '''To be implemented in subclasses.

        Called during destruction of the DAG. Subclasses should use this to stop 
        worker processes, if they used some.
        '''
        pass

    def get_spec(self):
        '''To be implemented in subclasses.
        '''
        raise NotImplementedError("Class %s does not implement 'get_spec'"%type(self).__name__)

    def request_batch(self, request):

        logger.debug("%s got request %s"%(type(self).__name__,request))

        upstream_request = copy.deepcopy(request)
        batch = self.provide(upstream_request)

        for (volume_type,roi) in request.volumes.items():
            assert volume_type in batch.volumes, "%s requested, but %s did not provide it."%(volume_type,type(self).__name__)
            volume = batch.volumes[volume_type]
            assert volume.roi == roi, "%s ROI %s requested, but ROI %s provided by %s."%(
                    volume_type,
                    roi,
                    volume.roi,
                    type(self).__name__
            )
            # ensure that the spatial dimensions are the same (other dimensions 
            # on top are okay, e.g., for affinities)
            dims = len(roi.get_shape())
            assert volume.data.shape[-dims:] == roi.get_shape(), "%s ROI %s requested, but shape of volume is %s provided by %s."%(
                    volume_type,
                    roi,
                    volume.data.shape,
                    type(self).__name__
            )

        logger.debug("%s provides %s"%(type(self).__name__,batch))

        return batch

    def provide(self, request):
        '''To be implemented in subclasses.

        Called with a batch request. Should return the requested batch.
        '''
        raise NotImplementedError("Class %s does not implement 'provide'"%type(self).__name__)
