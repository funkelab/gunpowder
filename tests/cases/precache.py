import time
from gunpowder import *
from .provider_test import ProviderTest

class Delay(BatchFilter):

    def prepare(self, request):
        time.sleep(1)

    def process(self, batch, request):
        pass

class TestPreCache(ProviderTest):

    def test_output(self):

        pipeline = self.test_source + Delay() + PreCache(num_workers=100, cache_size=100)

        with build(pipeline):

            start = time.time()

            for _ in range(100):
                batch = pipeline.request_batch(self.test_request)
                self.assertTrue(
                    batch.arrays[ArrayKeys.RAW].spec.roi ==
                    self.test_request[ArrayKeys.RAW].roi)

            # should be done in a bit more than 1 seconds, certainly much less
            # than 100
            self.assertTrue(time.time() - start < 10)

            # change request
            self.test_request[ArrayKeys.RAW].roi = \
                self.test_request[ArrayKeys.RAW].roi.shift((1,1,1))

            start = time.time()

            for _ in range(100):
                batch = pipeline.request_batch(self.test_request)
                self.assertTrue(
                    batch.arrays[ArrayKeys.RAW].spec.roi ==
                    self.test_request[ArrayKeys.RAW].roi)

            # should be done in a bit more than 1 seconds
            self.assertTrue(time.time() - start < 2)
