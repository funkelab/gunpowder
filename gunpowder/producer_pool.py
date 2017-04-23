import time
import multiprocessing
import Queue
import os
import sys

import logging
logger = logging.getLogger(__name__)

class NoResult(Exception):
    pass

class WorkersDied(Exception):
    pass

class ProducerPool(object):

    def __init__(self, callables, queue_size=10):
        self.watch_dog = multiprocessing.Process(target=self.__run_watch_dog, args=(callables,))
        self.__stop = multiprocessing.Event()
        self.__result_queue = multiprocessing.Queue(queue_size)

    def __del__(self):
        self.stop()

    def start(self):
        '''Start the pool of producers.'''

        if self.watch_dog.is_alive():
            logger.warning("trying to start workers, but they are already running")
            return

        self.__stop.clear()
        self.watch_dog.start()

    def get(self, timeout=0):
        '''Return the next result from the producer pool.

        If timeout is set and there is not result after the given number of 
        seconds, exception NoResult is raised.
        '''

        block = False
        if timeout == 0:
            timeout = 1
            block = True

        item = None
        while item == None:

            if not self.alive():
                raise WorkersDied()

            try:
                item = self.__result_queue.get(timeout=timeout)
            except Queue.Empty:
                # logger.debug("queue is still empty")
                if not block:
                    raise NoResult()

        if isinstance(item, Exception):
            raise item
        return item

    def stop(self):
        '''Stop the pool of producers.

        Items currently being produced will not be waited for and be discarded.'''

        self.__stop.set()
        self.watch_dog.join()

    def alive(self):
        '''Test if the pool is alive (i.e., all workers are running).
        '''
        return self.watch_dog.is_alive()

    def __run_watch_dog(self, callables):

        parent_pid = os.getppid()

        logger.debug("watchdog started with PID " + str(os.getpid()))
        logger.debug("parent PID " + str(parent_pid))

        workers = [ multiprocessing.Process(target=self.__run_worker, args=(c,)) for c in callables ]

        logger.debug("starting %d workers"%len(workers))
        for worker in workers:
            worker.start()

        while not self.__stop.wait(1):
            if os.getppid() != parent_pid:
                logger.error("parent of producer pool died, shutting down")
                break
            if not self.__all_workers_alive(workers):
                logger.error("at least one of my workers died, shutting down")
                break

        logger.info("terminating workers...")
        for worker in workers:
            worker.terminate()

        logger.info("joining workers...")
        for worker in workers:
            worker.join()

        logger.info("done")

    def __run_worker(self, target):

        parent_pid = os.getppid()

        logger.debug("worker started with PID " + str(os.getpid()))
        logger.debug("parent PID " + str(parent_pid))

        stop = False
        while not stop:

            result = None

            try:
                result = target()
            except Exception as e:
                result = e
                stop = True
            except:
                logger.error(sys.exc_info()[0])
                stop = True

            while result is not None:

                if os.getppid() != parent_pid:
                    logger.debug("worker %d: watch-dog died, stopping"%os.getpid())
                    stop = True
                    break

                try:
                    self.__result_queue.put(result, timeout=1)
                    result = None
                except Queue.Full:
                    logger.debug("worker %d: result queue is full, waiting to place my result"%os.getpid())

    def __all_workers_alive(self, workers):
        return all([ worker.is_alive() for worker in workers ])
