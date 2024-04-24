import time

import numpy as np
import pytest

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    BatchFilter,
    BatchRequest,
    Coordinate,
    PreCache,
    Roi,
    build,
)

from .helper_sources import ArraySource


class Delay(BatchFilter):
    def __init__(self, delay: float = 1):
        self.delay = delay

    def prepare(self, request):
        time.sleep(self.delay)
        return request

    def process(self, batch, request):
        pass


@pytest.mark.xfail(reason="Speedup is often dependent on hardware")
def test_speedup():
    delay = 0.2
    n_requests = 16
    a_workers = 2
    b_workers = 8
    perfect_speedup = a_workers / b_workers * delay * n_requests

    raw_key = ArrayKey("RAW")
    raw_array = Array(
        np.zeros([100, 100, 100], dtype=np.uint8),
        ArraySpec(
            roi=Roi((0, 0, 0), (100, 100, 100)),
            voxel_size=Coordinate((1, 1, 1)),
            dtype=np.uint8,
            interpolatable=True,
        ),
    )
    test_source = ArraySource(raw_key, raw_array)
    pipeline_a = test_source + Delay() + PreCache(num_workers=a_workers)
    pipeline_b = test_source + Delay() + PreCache(num_workers=b_workers)

    test_request = BatchRequest()
    test_request[raw_key] = ArraySpec(roi=Roi((20, 20, 20), (10, 10, 10)))

    with build(pipeline_a):
        start = time.time()

        for _ in range(n_requests):
            batch = pipeline_a.request_batch(test_request)
            assert batch.arrays[raw_key].spec.roi == test_request[raw_key].roi

        # should be done in a bit more than 4 seconds, certainly less than 8
        t_a_1 = time.time() - start

        # change request
        test_request[raw_key].roi = test_request[raw_key].roi.shift(Coordinate(1, 1, 1))

        start = time.time()

        for _ in range(n_requests):
            batch = pipeline_a.request_batch(test_request)
            assert batch.arrays[raw_key].spec.roi == test_request[raw_key].roi

        # should be done in a bit more than 4 seconds
        t_a_2 = time.time() - start

    with build(pipeline_b):
        start = time.time()

        for _ in range(n_requests):
            batch = pipeline_b.request_batch(test_request)
            assert batch.arrays[raw_key].spec.roi == test_request[raw_key].roi

        # should be done in a bit more than 4 seconds, certainly less than 8
        t_b_1 = time.time() - start

        # change request
        test_request[raw_key].roi = test_request[raw_key].roi.shift(Coordinate(1, 1, 1))

        start = time.time()

        for _ in range(n_requests):
            batch = pipeline_b.request_batch(test_request)
            assert batch.arrays[raw_key].spec.roi == test_request[raw_key].roi

        # should be done in a bit more than 4 seconds
        t_b_2 = time.time() - start

    assert t_a_1 - t_b_1 > perfect_speedup / 2, (t_a_1 - t_b_1, perfect_speedup)
    assert t_a_2 - t_b_2 > perfect_speedup / 2, (t_a_2 - t_b_2, perfect_speedup)
