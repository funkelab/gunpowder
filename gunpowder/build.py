import logging

logger = logging.getLogger(__name__)

class build(object):

    def __init__(self, batch_provider):
        self.batch_provider = batch_provider

    def __enter__(self):
        try:
            self.batch_provider.setup()
        except:
            logger.error("something went wrong during the setup of the pipeline, calling tear down")
            self.batch_provider.internal_teardown()
            logger.debug("tear down completed")
            raise
        return self.batch_provider

    def __exit__(self, type, value, traceback):
        logger.debug("leaving context, tearing down pipeline")
        self.batch_provider.internal_teardown()
        logger.debug("tear down completed")
