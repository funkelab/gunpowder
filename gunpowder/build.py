import logging

logger = logging.getLogger(__name__)

class build(object):

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __enter__(self):
        try:
            self.pipeline.setup()
        except:
            logger.error("something went wrong during the setup of the pipeline, calling tear down")
            self.pipeline.internal_teardown()
            logger.debug("tear down completed")
            raise
        return self.pipeline

    def __exit__(self, type, value, traceback):
        logger.debug("leaving context, tearing down pipeline")
        self.pipeline.internal_teardown()
        logger.debug("tear down completed")
