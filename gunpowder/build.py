class build(object):

    def __init__(self, batch_provider):
        self.batch_provider = batch_provider

    def __enter__(self):
        try:
            self.batch_provider.setup()
        except:
            self.batch_provider.teardown()
            raise
        return self.batch_provider

    def __exit__(self, type, value, traceback):
        self.batch_provider.teardown()
