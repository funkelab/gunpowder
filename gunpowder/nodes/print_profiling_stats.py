from batch_filter import BatchFilter

import logging
logger = logging.getLogger(__name__)

class PrintProfilingStats(BatchFilter):

    def process(self, batch):
        logger.info(batch.profiling_stats)
