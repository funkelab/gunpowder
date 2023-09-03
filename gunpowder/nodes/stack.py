from .batch_filter import BatchFilter
from gunpowder.array import Array
from gunpowder.batch import Batch
from gunpowder.profiling import Timing
import numpy as np
import random


class Stack(BatchFilter):
    """Request several batches and stack them together, introducing a new
    dimension for each array. This is useful to create batches with several
    samples and only makes sense if there is a source of randomness upstream.

    This node stacks only arrays, not points. The resulting batch will have the
    same point sets as found in the first batch requested upstream.

    Args:

        num_repetitions (``int``):

            How many upstream batches to stack.
    """

    def __init__(self, num_repetitions):
        self.num_repetitions = num_repetitions

    def provide(self, request):

        batches = []
        for _ in range(self.num_repetitions):
            upstream_request = request.copy()
            if upstream_request.is_deterministic():
                # if the request is deterministic, create new seeds for each
                # upstream request (otherwise we would get the same batch over
                # and over). Using randint here is still deterministic, since
                # the RNG was already seeded with the requests original seed.
                seed = random.randint(0, 2**32)
                upstream_request._random_seed = seed
            batch = self.get_upstream_provider().request_batch(upstream_request)
            batches.append(batch)

        timing = Timing(self)
        timing.start()

        batch = Batch()
        for b in batches:
            batch.profiling_stats.merge_with(b.profiling_stats)

        for key, spec in request.array_specs.items():
            data = np.stack([b[key].data for b in batches])
            batch[key] = Array(data, batches[0][key].spec.copy())

        # copy points of first batch requested
        for key, spec in request.graph_specs.items():
            batch[key] = batches[0][key]

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
