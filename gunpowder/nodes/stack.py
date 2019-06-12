from .batch_filter import BatchFilter
from gunpowder.array import Array
from gunpowder.batch import Batch
from gunpowder.profiling import Timing
import numpy as np

class Stack(BatchFilter):
    '''Request several batches and stack them together, introducing a new
    dimension for each array. This is useful to create batches with several
    samples and only makes sense if there is a source of randomness upstream.

    This node only supports batches containing arrays, not points.

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
            self.get_upstream_provider().provide(request)
            for _ in range(self.num_repetitions)
        ]

        batch = Batch()
        for key, spec in request.items():

            data = np.stack([b[key].data for b in batches])
            batch[key] = Array(
                data,
                batches[0][key].spec.copy())

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
