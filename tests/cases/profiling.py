import time

import numpy as np

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    BatchFilter,
    BatchRequest,
    PrintProfilingStats,
    Roi,
    build,
)

from .helper_sources import ArraySource


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


def test_profiling():
    raw_key = ArrayKey("RAW")
    raw_data = np.random.rand(100, 100, 100)
    raw_spec = ArraySpec(Roi((0, 0, 0), (100, 100, 100)), voxel_size=(1, 1, 1))
    raw_array = Array(raw_data, raw_spec)
    raw_source = ArraySource(raw_key, raw_array)
    pipeline = (
        raw_source
        + DelayNode(0.1, 0.2)
        + PrintProfilingStats(every=2)
        + DelayNode(0.2, 0.3)
    )

    request = BatchRequest()
    request.add(raw_key, (100, 100, 100))

    with build(pipeline):
        for i in range(5):
            batch = pipeline.request_batch(request)

    profiling_stats = batch.profiling_stats

    summary = profiling_stats.get_timing_summary("DelayNode", "prepare")

    # is the timing for each pass correct?
    assert summary.min() >= 0.1
    assert summary.min() <= 0.2 + 0.1  # bit of tolerance

    summary = profiling_stats.get_timing_summary("DelayNode", "process")

    assert summary.min() >= 0.2
    assert summary.min() <= 0.3 + 0.1  # bit of tolerance

    # is the upstream time correct?
    assert (
        profiling_stats.span_time() >= 0.1 + 0.2 + 0.2 + 0.3
    )  # total time spend upstream
    assert (
        profiling_stats.span_time() <= 0.1 + 0.2 + 0.2 + 0.3 + 0.1
    )  # plus bit of tolerance
