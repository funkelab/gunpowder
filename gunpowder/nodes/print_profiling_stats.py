import logging

from .batch_filter import BatchFilter
from gunpowder.profiling import ProfilingStats

logger = logging.getLogger(__name__)

class PrintProfilingStats(BatchFilter):

    def __init__(self, every=1):

        self.every = every
        self.n = 0
        self.accumulated_stats = ProfilingStats()

    def process(self, batch, request):

        self.n += 1
        print_stats = self.n%self.every == 0

        self.accumulated_stats.merge_with(batch.profiling_stats)

        if not print_stats:
            return

        stats = "\n"
        stats += "Profiling Stats\n"
        stats += "===============\n"
        stats += "\n"
        stats += str(self.accumulated_stats)
        stats += "\n"
        logger.info(stats)

        self.accumulated_stats = ProfilingStats()
