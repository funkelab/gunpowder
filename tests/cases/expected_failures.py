import gunpowder as gp
from gunpowder.nodes.batch_provider import BatchRequestError
from .helper_sources import ArraySource

from funlib.geometry import Coordinate

import numpy as np
import pytest


@pytest.mark.xfail()
def test_request_too_large():
    raw_key = gp.ArrayKey("RAW")
    data = np.ones((10, 10))
    array = gp.Array(
        data, gp.ArraySpec(voxel_size=Coordinate(100, 100), interpolatable=True)
    )
    source = ArraySource(raw_key, array)

    pipeline = (
        source
        + gp.Normalize(raw_key)
        + gp.RandomLocation()
        + gp.IntensityAugment(
            raw_key, scale_min=0, scale_max=0, shift_min=0.5, shift_max=0.5
        )
    )

    request = gp.BatchRequest()
    request.add(raw_key, (10000, 10000))

    with gp.build(pipeline):
        for i in range(100):
            with pytest.raises(BatchRequestError):
                batch = pipeline.request_batch(request)

            x = batch.arrays[raw_key].data
