import logging

from .batch_filter import BatchFilter
from gunpowder.profiling import Timing, ProfilingStats

logger = logging.getLogger(__name__)

class PrintProfilingStats(BatchFilter):

    def __init__(self, every=1):

        self.every = every
        self.n = 0
        self.accumulated_stats = ProfilingStats()
        self.__upstream_timing = Timing(self)
        self.__downstream_timing = Timing(self)

    def prepare(self, request):
        self.__upstream_timing.start()
        self.__downstream_timing.stop()

    def process(self, batch, request):
        self.__upstream_timing.stop()
        self.__downstream_timing.start()

        self.n += 1
        print_stats = self.n%self.every == 0

        self.accumulated_stats.merge_with(batch.profiling_stats)

        if not print_stats:
            return

        total_upstream_time = self.__upstream_timing.elapsed()
        self.__upstream_timing = Timing(self)

        span_start, span_end = self.accumulated_stats.span()

        stats = "\n"
        stats += "Profiling Stats\n"
        stats += "===============\n"
        stats += "\n"
        stats += str(self.accumulated_stats)
        stats += "Time span profiled   : %.2f\n"%self.accumulated_stats.span_time()
        stats += "Time spent upstream  : %.2f\n"%total_upstream_time
        stats += "\n"
        logger.info(stats)

        self.accumulated_stats = ProfilingStats()
