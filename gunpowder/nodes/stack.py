from .batch_filter import BatchFilter
from gunpowder.array import Array
from gunpowder.batch import Batch
from gunpowder.profiling import Timing
import numpy as np

class Stack(BatchFilter):
    '''Request several batches and stack them together, introducing a new
    dimension for each array. This is useful to create batches with several
    samples and only makes sense if there is a source of randomness upstream.

    This node stacks only arrays, not points. The resulting batch will have the
    same point sets as found in the first batch requested upstream.

    Args:

        num_repetitions (``int``):

            How many upstream batches to stack.
    '''

    def __init__(self, num_repetitions):

        self.num_repetitions = num_repetitions

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batches = [
            self.get_upstream_provider().request_batch(request)
            for _ in range(self.num_repetitions)
        ]

        batch = Batch()
        for key, spec in request.array_specs.items():

            data = np.stack([b[key].data for b in batches])
            batch[key] = Array(
                data,
                batches[0][key].spec.copy())

        # copy points of first batch requested
        for key, spec in request.points_specs.items():
            batch[key] = batches[0][key]

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
