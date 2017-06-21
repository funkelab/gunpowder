import logging

from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class PrintProfilingStats(BatchFilter):

    def process(self, batch, request):
        stats = "\n"
        stats += "Profiling Stats\n"
        stats += "===============\n"
        stats += "\n"
        stats += str(batch.profiling_stats)
        stats += "\n"
        logger.info(stats)
