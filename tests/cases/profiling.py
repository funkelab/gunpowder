from .provider_test import ProviderTest
from gunpowder import *
import time

class DelayNode(BatchFilter):

    def __init__(self, time_prepare, time_process):
        self.time_prepare = time_prepare
        self.time_process = time_process

    def prepare(self, request):
        time.sleep(self.time_prepare)

    def process(self, batch, request):
        time.sleep(self.time_process)

class TestProfiling(ProviderTest):

    def test_profiling(self):

        set_verbose(False)

        pipeline = (
                self.test_source +
                DelayNode(0.1, 0.2) +
                DelayNode(0.2, 0.3) +
                PrintProfilingStats()
        )

        with build(pipeline):
            for i in range(2):
                batch = pipeline.request_batch(self.test_request)

        profiling_stats = batch.profiling_stats

        for timing in profiling_stats.get_timings('DelayNode', 'prepare'):

            self.assertTrue('DelayNode' in timing.get_node_name())
            self.assertTrue('prepare' in timing.get_method_name())

            # is the timing for each pass correct?
            self.assertGreaterEqual(timing.elapsed(), 0.1)
            self.assertLessEqual(timing.elapsed(), 0.2 + 0.1) # bit of tolerance

        for timing in profiling_stats.get_timings('DelayNode', 'process'):

            self.assertTrue('DelayNode' in timing.get_node_name())
            self.assertTrue('process' in timing.get_method_name())

            self.assertGreaterEqual(timing.elapsed(), 0.2)
            self.assertLessEqual(timing.elapsed(), 0.3 + 0.1) # bit of tolerance

        # is the upstream time correct?
        self.assertGreaterEqual(profiling_stats.span_time(), 0.1+0.2+0.2+0.3) # total time spend upstream
        self.assertLessEqual(profiling_stats.span_time(), 0.1+0.2+0.2+0.3 + 0.1) # plus bit of tolerance
