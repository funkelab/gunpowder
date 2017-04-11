class BatchProvider(object):

    def __init__(self):
        self.upstream_providers = []

    def initialize_all(self):
        for p in self.upstream_providers:
            p.initialize_all()
        self.initialize()

    def add_upstream_provider(self, provider):
        self.upstream_providers.append(provider)

    def get_upstream_providers(self):
        return self.upstream_providers

    def initialize(self):
        '''To be implemented in subclasses.

        Called during initialization of the DAG. Callees can assume that all 
        upstream providers are initialized already.
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
