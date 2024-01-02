from .helper_sources import ArraySource, GraphSource
from gunpowder import (
    BatchRequest,
    Roi,
    Coordinate,
    GraphSpec,
    Array,
    ArrayKey,
    ArraySpec,
    RasterizeGraph,
    MergeProvider,
    RasterizationSettings,
    build,
)
from gunpowder.graph import GraphKey, Graph, Node, Edge

import numpy as np


def test_3d():
    graph_key = GraphKey("TEST_GRAPH")
    array_key = ArrayKey("TEST_ARRAY")
    rasterized_key = ArrayKey("RASTERIZED_ARRAY")
    voxel_size = Coordinate((40, 4, 4))

    graph = Graph(
        [
            # corners
            Node(id=1, location=np.array((-200, -200, -200))),
            Node(id=2, location=np.array((-200, -200, 199))),
            Node(id=3, location=np.array((-200, 199, -200))),
            Node(id=4, location=np.array((-200, 199, 199))),
            Node(id=5, location=np.array((199, -200, -200))),
            Node(id=6, location=np.array((199, -200, 199))),
            Node(id=7, location=np.array((199, 199, -200))),
            Node(id=8, location=np.array((199, 199, 199))),
            # center
            Node(id=9, location=np.array((0, 0, 0))),
            Node(id=10, location=np.array((-1, -1, -1))),
        ],
        [],
        GraphSpec(roi=Roi((-100, -100, -100), (300, 300, 300))),
    )

    array = Array(
        np.ones((10, 100, 100)),
        ArraySpec(
            roi=Roi((-200, -200, -200), (400, 400, 400)),
            voxel_size=voxel_size,
        ),
    )

    pipeline = (
        (GraphSource(graph_key, graph), ArraySource(array_key, array))
        + MergeProvider()
        + RasterizeGraph(
            graph_key,
            rasterized_key,
            ArraySpec(voxel_size=(40, 4, 4)),
        )
    )

    with build(pipeline):
        request = BatchRequest()
        roi = Roi((0, 0, 0), (200, 200, 200))

        request[graph_key] = GraphSpec(roi=roi)
        request[array_key] = ArraySpec(roi=roi)
        request[rasterized_key] = ArraySpec(roi=roi)

        batch = pipeline.request_batch(request)

        rasterized = batch.arrays[rasterized_key].data
        assert rasterized[0, 0, 0] == 1
        assert rasterized[2, 20, 20] == 0
        assert rasterized[4, 49, 49] == 1

    # same with different foreground/background labels

    pipeline = (
        (GraphSource(graph_key, graph), ArraySource(array_key, array))
        + MergeProvider()
        + RasterizeGraph(
            graph_key,
            rasterized_key,
            ArraySpec(voxel_size=(40, 4, 4)),
            RasterizationSettings(radius=1, fg_value=0, bg_value=1),
        )
    )

    with build(pipeline):
        request = BatchRequest()
        roi = Roi((0, 0, 0), (200, 200, 200))

        request[graph_key] = GraphSpec(roi=roi)
        request[array_key] = ArraySpec(roi=roi)
        request[rasterized_key] = ArraySpec(roi=roi)

        batch = pipeline.request_batch(request)

        rasterized = batch.arrays[rasterized_key].data
        assert rasterized[0, 0, 0] == 0
        assert rasterized[2, 20, 20] == 1
        assert rasterized[4, 49, 49] == 0

    # same with different radius and inner radius

    pipeline = (
        (GraphSource(graph_key, graph), ArraySource(array_key, array))
        + MergeProvider()
        + RasterizeGraph(
            graph_key,
            rasterized_key,
            ArraySpec(voxel_size=(40, 4, 4)),
            RasterizationSettings(
                radius=40, inner_radius_fraction=0.25, fg_value=1, bg_value=0
            ),
        )
    )

    with build(pipeline):
        request = BatchRequest()
        roi = Roi((0, 0, 0), (200, 200, 200))

        request[graph_key] = GraphSpec(roi=roi)
        request[array_key] = ArraySpec(roi=roi)
        request[rasterized_key] = ArraySpec(roi=roi)

        batch = pipeline.request_batch(request)

        rasterized = batch.arrays[rasterized_key].data

        # in the middle of the ball, there should be 0 (since inner radius is set)
        assert rasterized[0, 0, 0] == 0
        # check larger radius: rasterized point (0, 0, 0) should extend in
        # x,y by 10; z, by 1
        assert rasterized[0, 10, 0] == 1
        assert rasterized[0, 0, 10] == 1
        assert rasterized[1, 0, 0] == 1

        assert rasterized[2, 20, 20] == 0
        assert rasterized[4, 49, 49] == 0

    # same with different foreground/background labels
    # and GT_LABELS as mask of type np.uint64. Issue #193

    pipeline = (
        (GraphSource(graph_key, graph), ArraySource(array_key, array))
        + MergeProvider()
        + RasterizeGraph(
            graph_key,
            rasterized_key,
            ArraySpec(voxel_size=(40, 4, 4)),
            RasterizationSettings(radius=1, fg_value=0, bg_value=1, mask=array_key),
        )
    )

    with build(pipeline):
        request = BatchRequest()
        roi = Roi((0, 0, 0), (200, 200, 200))

        request[graph_key] = GraphSpec(roi=roi)
        request[array_key] = ArraySpec(roi=roi)
        request[rasterized_key] = ArraySpec(roi=roi)

        batch = pipeline.request_batch(request)

        rasterized = batch.arrays[rasterized_key].data
        assert rasterized[0, 0, 0] == 0
        assert rasterized[2, 20, 20] == 1
        assert rasterized[4, 49, 49] == 0

    # same with anisotropic radius

    pipeline = (
        (GraphSource(graph_key, graph), ArraySource(array_key, array))
        + MergeProvider()
        + RasterizeGraph(
            graph_key,
            rasterized_key,
            ArraySpec(voxel_size=(40, 4, 4)),
            RasterizationSettings(radius=(40, 40, 20), fg_value=1, bg_value=0),
        )
    )

    with build(pipeline):
        request = BatchRequest()
        roi = Roi((0, 0, 0), (120, 80, 80))

        request[graph_key] = GraphSpec(roi=roi)
        request[array_key] = ArraySpec(roi=roi)
        request[rasterized_key] = ArraySpec(roi=roi)

        batch = pipeline.request_batch(request)

        rasterized = batch.arrays[rasterized_key].data

        # check larger radius: rasterized point (0, 0, 0) should extend in
        # x,y by 10; z, by 1
        assert rasterized[0, 10, 0] == 1
        assert rasterized[0, 11, 0] == 0
        assert rasterized[0, 0, 5] == 1
        assert rasterized[0, 0, 6] == 0
        assert rasterized[1, 0, 0] == 1
        assert rasterized[2, 0, 0] == 0

    # same with anisotropic radius and inner radius

    pipeline = (
        (GraphSource(graph_key, graph), ArraySource(array_key, array))
        + MergeProvider()
        + RasterizeGraph(
            graph_key,
            rasterized_key,
            ArraySpec(voxel_size=(40, 4, 4)),
            RasterizationSettings(
                radius=(40, 40, 20), inner_radius_fraction=0.75, fg_value=1, bg_value=0
            ),
        )
    )

    with build(pipeline):
        request = BatchRequest()
        roi = Roi((0, 0, 0), (120, 80, 80))

        request[graph_key] = GraphSpec(roi=roi)
        request[array_key] = ArraySpec(roi=roi)
        request[rasterized_key] = ArraySpec(roi=roi)

        batch = pipeline.request_batch(request)

        rasterized = batch.arrays[rasterized_key].data

        # in the middle of the ball, there should be 0 (since inner radius is set)
        assert rasterized[0, 0, 0] == 0
        # check larger radius: rasterized point (0, 0, 0) should extend in
        # x,y by 10; z, by 1
        assert rasterized[0, 10, 0] == 1
        assert rasterized[0, 11, 0] == 0
        assert rasterized[0, 0, 5] == 1
        assert rasterized[0, 0, 6] == 0
        assert rasterized[1, 0, 0] == 1
        assert rasterized[2, 0, 0] == 0


def test_with_edge():
    graph_key = GraphKey("TEST_GRAPH")
    array_key = ArrayKey("TEST_ARRAY")
    rasterized_key = ArrayKey("RASTERIZED_ARRAY")
    voxel_size = Coordinate((40, 4, 4))

    array = Array(
        np.ones((10, 100, 100)),
        ArraySpec(
            roi=Roi((-200, -200, -200), (400, 400, 400)),
            voxel_size=voxel_size,
        ),
    )

    graph = Graph(
        [
            # corners
            Node(id=1, location=np.array((0, 4, 4))),
            Node(id=2, location=np.array((9, 4, 4))),
        ],
        [Edge(1, 2)],
        GraphSpec(roi=Roi((0, 0, 0), (10, 10, 10))),
    )

    pipeline = (
        (GraphSource(graph_key, graph), ArraySource(array_key, array))
        + MergeProvider()
        + RasterizeGraph(
            graph_key,
            rasterized_key,
            ArraySpec(voxel_size=(1, 1, 1)),
            settings=RasterizationSettings(0.5),
        )
    )

    with build(pipeline):
        request = BatchRequest()
        roi = Roi((0, 0, 0), (10, 10, 10))

        request[graph_key] = GraphSpec(roi=roi)
        request[rasterized_key] = ArraySpec(roi=roi)

        batch = pipeline.request_batch(request)

        rasterized = batch.arrays[rasterized_key].data

        assert (
            rasterized.sum() == 10
        ), f"rasterized has ones at: {np.where(rasterized==1)}"
