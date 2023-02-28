from .helper_sources import ArraySource
from gunpowder import (
    IntensityScaleShift,
    ArrayKey,
    build,
    Normalize,
    Array,
    ArraySpec,
    Roi,
    Coordinate,
    BatchRequest,
)

import numpy as np


def test_shift():

    raw_key = ArrayKey("RAW")
    array = Array(
        np.ones((10, 10)) * 2, ArraySpec(Roi((0, 0), (10, 10)), Coordinate(1, 1))
    )
    source = ArraySource(raw_key, array)

    request = BatchRequest()
    request.add(raw_key, Coordinate(10, 10))

    pipeline = source + Normalize(raw_key) + IntensityScaleShift(raw_key, 0.5, 10)

    with build(pipeline):
        batch = pipeline.request_batch(request)

        x = batch.arrays[raw_key].data
        assert np.isclose(x.min(), 10)
        assert np.isclose(x.max(), 10)
