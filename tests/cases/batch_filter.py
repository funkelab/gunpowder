from .helper_sources import ArraySource
from gunpowder import (
    ArrayKey,
    build,
    Array,
    ArraySpec,
    Roi,
    Coordinate,
    BatchRequest,
    BatchFilter,
)

import numpy as np
import random


class DummyNode(BatchFilter):
    def __init__(self, array, p=1.0):
        self.array = array
        self.p = p

    def skip_node(self, request):
        return random.random() > self.p

    def process(self, batch, request):
        batch[self.array].data = batch[self.array].data + 1


def test_skip():
    raw_key = ArrayKey("RAW")
    array = Array(
        np.ones((10, 10)),
        ArraySpec(Roi((0, 0), (10, 10)), Coordinate(1, 1)),
    )
    source = ArraySource(raw_key, array)

    request_1 = BatchRequest(random_seed=1)
    request_2 = BatchRequest(random_seed=2)

    request_1.add(raw_key, Coordinate(10, 10))
    request_2.add(raw_key, Coordinate(10, 10))

    pipeline = source + DummyNode(raw_key, p=0.5)

    with build(pipeline):
        batch_1 = pipeline.request_batch(request_1)
        batch_2 = pipeline.request_batch(request_2)

        x_1 = batch_1.arrays[raw_key].data
        x_2 = batch_2.arrays[raw_key].data

        assert x_1.max() == 2
        assert x_2.max() == 1
