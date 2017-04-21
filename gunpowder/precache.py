import multiprocessing
import Queue
import os
from batch_filter import BatchFilter
from batch_spec import BatchSpec
from profiling import Timing
from producer_pool import ProducerPool

import logging
logger = logging.getLogger(__name__)

class WorkersDiedException(Exception):
    pass

class PreCache(BatchFilter):

    def __init__(self, batch_spec_generator, cache_size=50, num_workers=20):
        '''
            batch_spec_generator:

                Callable that returns partial batch specs, to be passed 
                upstream.

            cache_size: int

                How many batches to pre-cache.

            num_workers: int

                How many processes to spawn to fill the cache.
        '''
        self.batch_spec_generator = batch_spec_generator
        self.batches = multiprocessing.Queue(maxsize=cache_size)
        self.workers = ProducerPool([ lambda i=i: self.__run_worker(i) for i in range(num_workers) ], queue_size=cache_size)

    def setup(self):
        self.workers.start()

    def teardown(self):
        self.workers.stop()

    def request_batch(self, batch_spec):

        timing = Timing(self)
        timing.start()

        logger.debug("getting batch from queue...")
        batch = self.workers.get()

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __run_worker(self, i):

        batch_spec = self.batch_spec_generator()
        return self.get_upstream_provider().request_batch(batch_spec)
