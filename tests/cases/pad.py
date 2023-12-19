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
def test_output(mode):
    array_key = ArrayKey("TEST_ARRAY")
    graph_key = GraphKey("TEST_GRAPH")

    array_spec = ArraySpec(
        roi=Roi((200, 20, 20), (1800, 180, 180)), voxel_size=(20, 2, 2)
    )
    roi_voxel = array_spec.roi / array_spec.voxel_size
    data = np.zeros(roi_voxel.shape, dtype=np.uint32)
    data[:, ::2] = 100
    array = Array(data, spec=array_spec)

    graph_spec = GraphSpec(roi=Roi((200, 20, 20), (1800, 180, 180)))
    graph = Graph([], [], graph_spec)

    source = (
        ArraySource(array_key, array),
        GraphSource(graph_key, graph),
    ) + MergeProvider()

    pipeline = (
        source
        + Pad(array_key, Coordinate((20, 20, 20)), value=1, mode=mode)
        + Pad(graph_key, Coordinate((10, 10, 10)), mode=mode)
    )

    with build(pipeline):
        assert pipeline.spec[array_key].roi == Roi((180, 0, 0), (1840, 220, 220))
        assert pipeline.spec[graph_key].roi == Roi((190, 10, 10), (1820, 200, 200))

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
