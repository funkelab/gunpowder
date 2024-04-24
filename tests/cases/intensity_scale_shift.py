import numpy as np

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    BatchRequest,
    Coordinate,
    IntensityScaleShift,
    Roi,
    build,
)

from .helper_sources import ArraySource


def test_shift():
    raw_key = ArrayKey("RAW")
    array = Array(
        np.ones((10, 10), dtype=np.float32) * 2,
        ArraySpec(Roi((0, 0), (10, 10)), Coordinate(1, 1)),
    )
    source = ArraySource(raw_key, array)

    request = BatchRequest()
    request.add(raw_key, Coordinate(10, 10))

    pipeline = source + IntensityScaleShift(raw_key, 0.5, 10)

    with build(pipeline):
        batch = pipeline.request_batch(request)

        x = batch.arrays[raw_key].data
        assert np.isclose(x.min(), 11)
        assert np.isclose(x.max(), 11)
