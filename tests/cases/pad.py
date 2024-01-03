from .helper_sources import ArraySource, GraphSource
from gunpowder import (
    BatchRequest,
    ArraySpec,
    Roi,
    Coordinate,
    Graph,
    GraphKey,
    GraphSpec,
    Array,
    ArrayKey,
    Pad,
    build,
    MergeProvider,
)

import pytest
import numpy as np

from itertools import product


@pytest.mark.parametrize("mode", ["constant", "reflect"])
def test_padding(mode):
    array_key = ArrayKey("TEST_ARRAY")
    graph_key = GraphKey("TEST_GRAPH")

    array_spec = ArraySpec(roi=Roi((200, 20, 20), (600, 60, 60)), voxel_size=(20, 2, 2))
    roi_voxel = array_spec.roi / array_spec.voxel_size
    data = np.zeros(roi_voxel.shape, dtype=np.uint32)
    data[:, ::2] = 100
    array = Array(data, spec=array_spec)

    graph_spec = GraphSpec(roi=Roi((200, 20, 20), (600, 60, 60)))
    graph = Graph([], [], graph_spec)

    source = (
        ArraySource(array_key, array),
        GraphSource(graph_key, graph),
    ) + MergeProvider()

    pipeline = (
        source
        + Pad(array_key, Coordinate((200, 20, 20)), value=1, mode=mode)
        + Pad(graph_key, Coordinate((100, 10, 10)), mode=mode)
    )

    with build(pipeline):
        assert pipeline.spec[array_key].roi == Roi((0, 0, 0), (1000, 100, 100))
        assert pipeline.spec[graph_key].roi == Roi((100, 10, 10), (800, 80, 80))

        batch = pipeline.request_batch(
            BatchRequest({array_key: ArraySpec(Roi((180, 0, 0), (40, 40, 40)))})
        )

        data = batch.arrays[array_key].data
        if mode == "constant":
            octants = [
                (1 * 10 * 10) if zi + yi + xi < 3 else 100 * 1 * 5 * 10
                for zi, yi, xi in product(range(2), range(2), range(2))
            ]
            assert np.sum(data) == np.sum(octants), (
                np.sum(data),
                np.sum(octants),
                np.unique(data),
            )
        elif mode == "reflect":
            octants = [100 * 1 * 5 * 10 for _ in range(8)]
            assert np.sum(data) == np.sum(octants), (
                np.sum(data),
                np.sum(octants),
                data,
            )

        # 1 x 10 x (10,30,10)
        batch = pipeline.request_batch(
            BatchRequest({array_key: ArraySpec(Roi((200, 20, 0), (20, 20, 100)))})
        )
        data = batch.arrays[array_key].data

        if mode == "constant":
            lower_pad = 1 * 10 * 10
            upper_pad = 1 * 10 * 10
            center = 100 * 1 * 5 * 30
            assert np.sum(data) == np.sum((lower_pad, upper_pad, center)), (
                np.sum(data),
                np.sum((lower_pad, upper_pad, center)),
            )
        elif mode == "reflect":
            lower_pad = 100 * 1 * 5 * 10
            upper_pad = 100 * 1 * 5 * 10
            center = 100 * 1 * 5 * 30
            assert np.sum(data) == np.sum((lower_pad, upper_pad, center)), (
                np.sum(data),
                np.sum((lower_pad, upper_pad, center)),
            )
