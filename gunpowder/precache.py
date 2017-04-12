import multiprocessing
import atexit
from batch_filter import BatchFilter
from batch_spec import BatchSpec

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

    def __del__(self):
        print("PreCache: being killed")
        for worker in self.workers:
            worker.terminate()

    def initialize(self):

        self.stopped = multiprocessing.Event()
        self.stopped.clear()
        for worker in self.workers:
            worker.start()
        atexit.register(self.__del__)

    def request_batch(self, batch_spec):
        print("PreCache: getting batch from queue...")
        batch = self.batches.get()
        print("PreCache: ...got it")
        return batch

    def __run_worker(self, i):

        import random
        random.seed(42)

        while not self.stopped.is_set():
            batch_spec = self.batch_spec_generator()
            batch = self.get_upstream_provider().request_batch(batch_spec)
            print("PreCache Worker %d: putting a batch in the queue..."%i)
            if self.stopped.is_set():
                return
            self.batches.put(batch)
            print("PreCache Worker %d: ...done"%i)
