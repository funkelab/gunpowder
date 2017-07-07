import logging

from .batch_filter import BatchFilter
from gunpowder.profiling import ProfilingStats

logger = logging.getLogger(__name__)

class PrintProfilingStats(BatchFilter):

    def __init__(self, every=1):

        self.every = every
        self.n = 0
        self.accumulated_stats = ProfilingStats()
        self.__last_span_end = 0

    def process(self, batch, request):

        self.n += 1
        print_stats = self.n%self.every == 0

        self.accumulated_stats.merge_with(batch.profiling_stats)

        if not print_stats:
            return

        span_start, span_end = self.accumulated_stats.span()

        stats = "\n"
        stats += "Profiling Stats\n"
        stats += "===============\n"
        stats += "\n"
        stats += str(self.accumulated_stats)
        if self.__last_span_end != 0:
            time_since_last_span = span_start - self.__last_span_end
            stats += "Time spend downstream          : %.2f\n"%time_since_last_span
        stats += "\n"
        logger.info(stats)

        self.accumulated_stats = ProfilingStats()
        self.__last_span_end = span_end
