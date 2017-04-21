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
        raise RuntimeError("Class %s does not implement 'get_spec'"%self.__class__)

    def request_batch(self, batch_spec):
        '''To be implemented in subclasses.
        '''
        raise RuntimeError("Class %s does not implement 'request_batch'"%self.__class__)
