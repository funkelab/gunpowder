import multiprocessing
import Queue
import atexit
from batch_filter import BatchFilter
from batch_spec import BatchSpec

import logging
logger = logging.getLogger(__name__)

class WorkersFailedException(Exception):
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
        self.workers = [ multiprocessing.Process(target=self.__run_worker, args=(i,)) for i in range(num_workers) ]
        self.stopped = None

    def __del__(self):
        self.stop_workers()

    def workers_alive(self):
        return all([worker.is_alive() for worker in self.workers])

    def stop_workers(self):

        logger.info("terminating workers...")
        for worker in self.workers:
            worker.terminate()
        for worker in self.workers:
            worker.join()

    def initialize(self):

        if self.stopped is None:
            logger.debug("PreCache: starting %d workers"%len(self.workers))
            self.stopped = multiprocessing.Event()
            self.stopped.clear()
            for worker in self.workers:
                worker.start()
            atexit.register(self.__del__)

    def request_batch(self, batch_spec):
        logger.debug("PreCache: getting batch from queue...")
        batch = None
        while batch is None:
            try:
                batch = self.batches.get(timeout=1)
            except Queue.Empty:
                logging.info("waiting for batch...")
            if not self.workers_alive():
                self.stop_workers()
                raise WorkersFailedException()
        logger.debug("PreCache: ...got it")
        return batch

    def __run_worker(self, i):

        while not self.stopped.is_set():
            logger.debug("PreCache Worker %d: requesting a batch..."%i)
            batch_spec = self.batch_spec_generator()
            batch = self.get_upstream_provider().request_batch(batch_spec)
            logger.debug("PreCache Worker %d: putting a batch in the queue..."%i)
            if self.stopped.is_set():
                return
            self.batches.put(batch)
            logger.debug("PreCache Worker %d: ...done"%i)
