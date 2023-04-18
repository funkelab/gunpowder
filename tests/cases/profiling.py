from .provider_test import ProviderTest
from gunpowder import *
import time


class DelayNode(BatchFilter):
    def __init__(self, time_prepare, time_process):
        self.time_prepare = time_prepare
        self.time_process = time_process

    def prepare(self, request):
        time.sleep(self.time_prepare)

        deps = request
        return deps

    def process(self, batch, request):
        time.sleep(self.time_process)


class TestProfiling(ProviderTest):
    def test_profiling(self):
        pipeline = (
            self.test_source
            + DelayNode(0.1, 0.2)
            + PrintProfilingStats(every=2)
            + DelayNode(0.2, 0.3)
        )

        with build(pipeline):
            for i in range(5):
                batch = pipeline.request_batch(self.test_request)

        profiling_stats = batch.profiling_stats

        summary = profiling_stats.get_timing_summary("DelayNode", "prepare")

        # is the timing for each pass correct?
        self.assertGreaterEqual(summary.min(), 0.1)
        self.assertLessEqual(summary.min(), 0.2 + 0.1)  # bit of tolerance

        summary = profiling_stats.get_timing_summary("DelayNode", "process")

        self.assertGreaterEqual(summary.min(), 0.2)
        self.assertLessEqual(summary.min(), 0.3 + 0.1)  # bit of tolerance

        # is the upstream time correct?
        self.assertGreaterEqual(
            profiling_stats.span_time(), 0.1 + 0.2 + 0.2 + 0.3
        )  # total time spend upstream
        self.assertLessEqual(
            profiling_stats.span_time(), 0.1 + 0.2 + 0.2 + 0.3 + 0.1
        )  # plus bit of tolerance
