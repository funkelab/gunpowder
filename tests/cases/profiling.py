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

    set_verbose(False)

    def test_profiling(self):

        pipeline = (
                self.test_source +
                DelayNode(0.1, 0.2) +
                DelayNode(0.2, 0.3) +
                PrintProfilingStats() +
                DelayNode(0.3, 0.4)
        )

        with build(pipeline):
            for i in range(2):
                batch = pipeline.request_batch(self.test_request)

        profiling_stats = batch.profiling_stats

        for name, timing in profiling_stats.get_timings():

            self.assertTrue(timing.get_name() == name)

            # is the timing for each pass correct?
            if 'prepare' in name:
                self.assertTrue(timing.elapsed() >= 0.1)
                self.assertTrue(timing.elapsed() <= 0.2 + 0.1) # bit of tolerance
            elif 'process' in name:
                self.assertTrue(timing.elapsed() >= 0.2)
                self.assertTrue(timing.elapsed() <= 0.3 + 0.1) # bit of tolerance
            else:
                self.assertTrue(False)

        # is the upstream time correct?
        self.assertAlmostEqual(profiling_stats.upstream_total(), 0.1+0.2+0.2+0.3) # total time spend upstream

        # is the downstream time correct?
        self.assertAlmostEqual(profiling_stats.downstream_total(), 0.3+0.4) # total time spend downstream
