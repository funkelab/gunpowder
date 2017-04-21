import multiprocessing
import Queue
import os
from batch_filter import BatchFilter
from batch_spec import BatchSpec
from profiling import Timing

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
        self.workers = [ multiprocessing.Process(target=self.__run_worker, args=(i,)) for i in range(num_workers) ]

    def setup(self):
        self.__start_workers()

    def teardown(self):
        print("PreCache: teardown called")
        self.__stop_workers()

    def request_batch(self, batch_spec):

        timing = Timing(self)
        timing.start()

        logger.debug("getting batch from queue...")
        batch = None
        while batch is None:

            try:

                batch = self.batches.get(timeout=1)

            except Queue.Empty:

                logging.info("waiting for batch...")

            if not self.__workers_alive():

                logger.error("at least one of my workers died")
                self.__stop_workers()
                raise WorkersDiedException()

        logger.debug("...got it")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __del__(self):
        self.__stop_workers()

    def __start_workers(self):

        if self.__workers_alive():
            logger.warning("trying to start workers, but they are already running")
            return

        logger.debug("starting %d workers"%len(self.workers))
        for worker in self.workers:
            worker.start()

    def __stop_workers(self):

        logger.info("terminating workers...")
        for worker in self.workers:
            worker.terminate()

        logger.info("joining workers...")
        for worker in self.workers:
            worker.join()

        logger.info("done")

    def __workers_alive(self):
        return all([worker.is_alive() for worker in self.workers])

    def __run_worker(self, i):

        while not self.__parent_died():

            logger.info("PreCache Worker %d: requesting a batch..."%i)

            batch_spec = self.batch_spec_generator()
            batch = self.get_upstream_provider().request_batch(batch_spec)

            logger.info("PreCache Worker %d: putting a batch in the queue..."%i)

            while not self.__parent_died():

                try:
                    self.batches.put(batch, timeout=1)
                    logger.info("PreCache Worker %d: ...done"%i)
                    break
                except Queue.Full:
                    logger.info("PreCache Worker %d: ...queue is already full..."%i)

        logger.info("PreCache Worker %d: parent died, exiting..."%i)

    def __parent_died(self):
        return os.getppid() == 1
