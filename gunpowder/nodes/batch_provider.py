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
        raise NotImplementedError("Class %s does not implement 'get_spec'"%self.__class__)

    def request_batch(self, request):

        logger.debug("%s got request %s"%(self.__class__,request))

        upstream_request = copy.deepcopy(request)
        return self.provide(upstream_request)

    def provide(self, request):
        '''To be implemented in subclasses.

        Called with a batch request. Should return the requested batch.
        '''
        raise NotImplementedError("Class %s does not implement 'provide'"%self.__class__)

    @property
    def resolution(self):
        '''To be implemented in subclasses.
        '''
        raise NotImplementedError("Class %s does not implement 'resolution'" % self.__class__)
