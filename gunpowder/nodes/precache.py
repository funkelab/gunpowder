import copy
import logging
import multiprocessing

from .batch_filter import BatchFilter
from gunpowder.profiling import Timing
from gunpowder.producer_pool import ProducerPool

logger = logging.getLogger(__name__)

class WorkersDiedException(Exception):
    pass

class PreCache(BatchFilter):
    '''Pre-cache repeated equal batch requests. For the first of a series of
    equal batch request, a set of workers is spawned to pre-cache the batches
    in parallel processes. This way, subsequent requests can be served quickly.

    This node only makes sense if:

    1. Incoming batch requests are repeatedly the same.
    2. There is a source of randomness in upstream nodes.

    Args:

        cache_size (``int``):

            How many batches to hold at most in the cache.

        num_workers (``int``):

            How many processes to spawn to fill the cache.
    '''

    def __init__(self, cache_size=50, num_workers=20):

        self.current_request = None
        self.workers = None
        self.cache_size = cache_size
        self.num_workers = num_workers

    def teardown(self):

        if self.workers is not None:
            self.workers.stop()

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        if request != self.current_request:

            if self.workers is not None:
                logger.info("new request received, stopping current workers...")
                self.workers.stop()

            self.current_request = copy.deepcopy(request)

            logger.info("starting new set of workers...")
            self.workers = ProducerPool([ lambda i=i: self.__run_worker(i) for i in range(self.num_workers) ], queue_size=self.cache_size)
            self.workers.start()

        logger.debug("getting batch from queue...")
        batch = self.workers.get()

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __run_worker(self, i):

        return self.get_upstream_provider().request_batch(self.current_request)
