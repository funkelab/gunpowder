import logging

from .batch_filter import BatchFilter
from gunpowder.profiling import Timing, TimingSummary, ProfilingStats

logger = logging.getLogger(__name__)

class PrintProfilingStats(BatchFilter):
    '''Print profiling information about nodes upstream of this node in the DAG.

    The output also includes a ``TOTAL`` section, which shows the wall-time 
    spent in the upstream and downstream passes. For the downstream pass, this 
    information is not available in the first iteration, since the request-batch 
    cycle is not completed, yet.

    Args:

        every (``int``):

            Collect statistics about that many batch requests and show min,
            max, mean, and median runtimes.
    '''

    def __init__(self, every=1):

        self.every = every
        self.n = 0
        self.accumulated_stats = ProfilingStats()
        self.__upstream_timing = Timing(self)
        self.__upstream_timing_summary = TimingSummary()
        self.__downstream_timing = Timing(self)
        self.__downstream_timing_summary = TimingSummary()

    def prepare(self, request):

        self.__downstream_timing.stop()
        # skip the first one, where we don't know how much time we spent 
        # downstream
        if self.__downstream_timing.elapsed() > 0:
            self.__downstream_timing_summary.add(self.__downstream_timing)
            self.__downstream_timing = Timing(self)

        self.__upstream_timing.start()

    def process(self, batch, request):

        self.__upstream_timing.stop()
        self.__upstream_timing_summary.add(self.__upstream_timing)
        self.__upstream_timing = Timing(self)

        self.__downstream_timing.start()

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
        stats += "NODE".ljust(20)
        stats += "METHOD".ljust(10)
        stats += "COUNTS".ljust(10)
        stats += "MIN".ljust(10)
        stats += "MAX".ljust(10)
        stats += "MEAN".ljust(10)
        stats += "MEDIAN".ljust(10)
        stats += "\n"

        summaries = self.accumulated_stats.get_timing_summaries().items()
        summaries.sort()

        for (node_name, method_name), summary in summaries:

            if summary.counts() > 0:
                stats += node_name[:19].ljust(20)
                stats += method_name[:19].ljust(10) if method_name is not None else ' '*10
                stats += ("%d"%summary.counts())[:9].ljust(10)
                stats += ("%.2f"%summary.min())[:9].ljust(10)
                stats += ("%.2f"%summary.max())[:9].ljust(10)
                stats += ("%.2f"%summary.mean())[:9].ljust(10)
                stats += ("%.2f"%summary.median())[:9].ljust(10)
                stats += "\n"

        stats += "\n"
        stats += "TOTAL"
        stats += "\n"

        for phase, summary in zip(['upstream', 'downstream'], [self.__upstream_timing_summary, self.__downstream_timing_summary]):

            if summary.counts() > 0:
                stats += phase[:19].ljust(30)
                stats += ("%d"%summary.counts())[:9].ljust(10)
                stats += ("%.2f"%summary.min())[:9].ljust(10)
                stats += ("%.2f"%summary.max())[:9].ljust(10)
                stats += ("%.2f"%summary.mean())[:9].ljust(10)
                stats += ("%.2f"%summary.median())[:9].ljust(10)
                stats += "\n"

        stats += "\n"

        logger.info(stats)

        # reset summaries
        self.accumulated_stats = ProfilingStats()
        self.__upstream_timing_summary = TimingSummary()
        self.__downstream_timing_summary = TimingSummary()
