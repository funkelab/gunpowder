try:
    import Queue
except:
    import queue as Queue
import logging
import multiprocessing
import os
import sys
import time
import traceback

logger = logging.getLogger(__name__)

class NoResult(Exception):
    pass

class ParentDied(Exception):
    pass

class WorkersDied(Exception):
    pass

class ProducerPool(object):

    def __init__(self, callables, queue_size=10):
        self.__watch_dog = multiprocessing.Process(target=self.__run_watch_dog, args=(callables,))
        self.__stop = multiprocessing.Event()
        self.__result_queue = multiprocessing.Queue(queue_size)

    def __del__(self):
        self.stop()

    def start(self):
        '''Start the pool of producers.'''

        if self.__watch_dog.is_alive():
            logger.warning("trying to start workers, but they are already running")
            return

        self.__stop.clear()
        self.__watch_dog.start()

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

            try:
                item = self.__result_queue.get(timeout=timeout)
            except Queue.Empty:
                if not block:
                    raise NoResult()

        if isinstance(item, Exception):
            raise item
        return item

    def stop(self):
        '''Stop the pool of producers.

        Items currently being produced will not be waited for and be discarded.'''

        self.__stop.set()
        self.__watch_dog.join()

    def __run_watch_dog(self, callables):

        parent_pid = os.getppid()

        logger.debug("watchdog started with PID " + str(os.getpid()))
        logger.debug("parent PID " + str(parent_pid))

        workers = [ multiprocessing.Process(target=self.__run_worker, args=(c,)) for c in callables ]

        try:

            logger.debug("starting %d workers"%len(workers))
            for worker in workers:
                worker.start()

            while not self.__stop.wait(1):
                if os.getppid() != parent_pid:
                    logger.error("parent of producer pool died, shutting down")
                    self.__result_queue.put(ParentDied())
                    break
                if not self.__all_workers_alive(workers):
                    logger.error("at least one of my workers died, shutting down")
                    self.__result_queue.put(WorkersDied())
                    break
        except:
            pass

        finally:

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

        result = None
        while True:

            if os.getppid() != parent_pid:
                logger.debug("worker %d: watch-dog died, stopping"%os.getpid())
                break

            if result is None:

                try:
                    result = target()
                except Exception as e:
                    result = e
                    traceback.print_exc()
                    # don't stop on normal exceptions -- place them in result queue 
                    # and let them be handled by caller
                except:
                    logger.error("received error: " + str(sys.exc_info()[0]))
                    # this is most likely a keyboard interrupt, stop process
                    break

            try:
                self.__result_queue.put(result, timeout=1)
                result = None
            except Queue.Full:
                logger.debug("worker %d: result queue is full, waiting to place my result"%os.getpid())

        logger.debug("worker with PID " + str(os.getpid()) + " exiting")
        os._exit(1)

    def __all_workers_alive(self, workers):
        return all([ worker.is_alive() for worker in workers ])
