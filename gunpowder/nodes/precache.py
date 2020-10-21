import copy
import logging
import multiprocessing

from .batch_filter import BatchFilter
from gunpowder.profiling import Timing
from gunpowder.producer_pool import ProducerPool

from collections import deque

logger = logging.getLogger(__name__)


class WorkersDiedException(Exception):
    pass

class PreCache(BatchFilter):
    '''Pre-cache repeated equal batch requests. For the first of a series of
    equal batch request, a set of workers is spawned to pre-cache the batches
    in parallel processes. This way, subsequent requests can be served quickly.

    A note on changing the requests sent to `PreCache`.
    Given requests A and B, if requests are sent in the sequence:
    A, ..., A, B, A, ..., A, B, A, ...
    Precache will build a Queue of batches that satisfy A, and handle requests
    B on demand. This prevents `PreCache` from discarding the queue on every
    SnapshotRequest.
    However if B request replace A as the most common request, i.e.:
    A, A, A, ..., A, B, B, B, ...,
    `PreCache` will discard the A queue and build a B queue after it has seen
    more B requests than A requests out of the last 5 requests.

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

        # keep track of recent requests
        self.last_5 = deque([None,] * 5, maxlen=5)

    def teardown(self):

        if self.workers is not None:
            self.workers.stop()

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        # update recent requests
        self.last_5.popleft()
        self.last_5.append(request)

        if request != self.current_request:

            current_count = sum(
                [
                    recent_request == self.current_request
                    for recent_request in self.last_5
                ]
            )
            new_count = sum(
                [recent_request == request for recent_request in self.last_5]
            )
            if new_count > current_count or self.current_request is None:

                if self.workers is not None:
                    logger.info("new request received, stopping current workers...")
                    self.workers.stop()

                self.current_request = copy.deepcopy(request)

                logger.info("starting new set of workers...")
                self.workers = ProducerPool(
                    [lambda i=i: self.__run_worker(i) for i in range(self.num_workers)],
                    queue_size=self.cache_size,
                )
                self.workers.start()

                logger.debug("getting batch from queue...")
                batch = self.workers.get()

                timing.stop()
                batch.profiling_stats.add(timing)

            else:
                logger.debug("Resolving new request sequentially")
                batch = self.get_upstream_provider().request_batch(request)

                timing.stop()
                batch.profiling_stats.add(timing)

        else:
            logger.debug("getting batch from queue...")
            batch = self.workers.get()

            timing.stop()
            batch.profiling_stats.add(timing)

        return batch

    def __run_worker(self, i):

        return self.get_upstream_provider().request_batch(self.current_request)
