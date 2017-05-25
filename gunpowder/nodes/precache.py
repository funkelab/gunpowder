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

    def __init__(self, request, cache_size=50, num_workers=20):
        '''
            request:

                A BatchRequest used to pre-cache batches.

            cache_size: int

                How many batches to pre-cache.

            num_workers: int

                How many processes to spawn to fill the cache.
        '''
        self.request = copy.deepcopy(request)
        self.batches = multiprocessing.Queue(maxsize=cache_size)
        self.workers = ProducerPool([ lambda i=i: self.__run_worker(i) for i in range(num_workers) ], queue_size=cache_size)

    def setup(self):
        self.workers.start()

    def teardown(self):
        self.workers.stop()

    def request_batch(self, request):

        timing = Timing(self)
        timing.start()

        logger.debug("getting batch from queue...")
        batch = self.workers.get()

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __run_worker(self, i):

        request = copy.deepcopy(self.request)
        return self.get_upstream_provider().request_batch(request)
