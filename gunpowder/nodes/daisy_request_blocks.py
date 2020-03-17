from gunpowder.batch import Batch
from gunpowder.ext import daisy
from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.roi import Roi
import logging
import multiprocessing
import time

logger = logging.getLogger(__name__)


class DaisyRequestBlocks(BatchFilter):
    '''Iteratively requests batches similar to ``reference`` from upstream
    providers, with their ROIs set to blocks distributed by ``daisy``.

    The ROIs of the array or point specs in the reference can be set to either
    the block's ``read_roi`` or ``write_roi``, see parameter ``roi_map``.

    The batch request to this node has to be empty, as there is no guarantee
    that this node will get to process all chunks required to fulfill a
    particular batch request.

    Args:

        reference (:class:`BatchRequest`):

            A reference :class:`BatchRequest`. This request will be shifted
            according to blocks distributed by ``daisy``.

        roi_map (``dict`` from :class:`ArrayKey` or :class:`GraphKey` to
        ``string``):

            A map indicating which daisy block ROI (``read_roi`` or
            ``write_roi``) to use for which item in the reference request.

        num_workers (``int``, optional):

            If set to >1, upstream requests are made in parallel with that
            number of workers.

        block_done_callback (function, optional):

            If given, will be called with arguments ``(block, start,
            duration)`` for each block that was processed. ``start`` and
            ``duration`` will be given in seconds, as in ``start =
            time.time()`` and ``duration = time.time() - start``, right before
            and after a block gets processed.

            This callback can be used to log blocks that have successfully
            finished processing, which can be used in ``check_function`` of
            ``daisy.run_blockwise`` to skip already processed blocks in
            repeated runs.
    '''

    def __init__(
            self,
            reference,
            roi_map,
            num_workers=1,
            block_done_callback=None):

        self.reference = reference
        self.roi_map = roi_map
        self.num_workers = num_workers
        self.block_done_callback = block_done_callback

        if num_workers > 1:
            self.request_queue = multiprocessing.Queue(maxsize=0)

    def provide(self, request):

        empty_request = (len(request) == 0)
        if not empty_request:
            raise RuntimeError(
                "requests made to DaisyRequestBlocks have to be empty")

        if self.num_workers > 1:

            self.workers = [
                multiprocessing.Process(target=self.__get_chunks)
                for _ in range(self.num_workers)
            ]

            for worker in self.workers:
                worker.start()

            for worker in self.workers:
                worker.join()

        else:

            self.__get_chunks()

        return Batch()

    def __get_chunks(self):

        daisy_client = daisy.Client()

        while True:

            block = daisy_client.acquire_block()

            if block is None:
                return

            logger.info("Processing block %s", block)
            start = time.time()

            chunk_request = self.reference.copy()

            for key, reference_spec in self.reference.items():

                roi_type = self.roi_map.get(key, None)

                if roi_type is None:
                    raise RuntimeError(
                        "roi_map does not map item %s to either 'read_roi' "
                        "or 'write_roi'" % key)

                if roi_type == 'read_roi':
                    chunk_request[key].roi = Roi(
                        block.read_roi.get_offset(),
                        block.read_roi.get_shape())
                elif roi_type == 'write_roi':
                    chunk_request[key].roi = Roi(
                        block.write_roi.get_offset(),
                        block.write_roi.get_shape())
                else:
                    raise RuntimeError(
                        "%s is not a vaid ROI type (read_roi or write_roi)")

            self.get_upstream_provider().request_batch(chunk_request)

            end = time.time()
            if self.block_done_callback:
                self.block_done_callback(block, start, end - start)

            daisy_client.release_block(block, ret=0)
