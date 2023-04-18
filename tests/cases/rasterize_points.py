from .provider_test import ProviderTest
from gunpowder import (
    BatchProvider,
    BatchRequest,
    Batch,
    Roi,
    Coordinate,
    GraphSpec,
    Array,
    ArrayKey,
    ArrayKeys,
    ArraySpec,
    RasterizeGraph,
    RasterizationSettings,
    build,
)
from gunpowder.graph import GraphKeys, GraphKey, Graph, Node, Edge

import numpy as np
import math
from random import randint


class GraphTestSource3D(BatchProvider):
    def __init__(self):
        self.voxel_size = Coordinate((40, 4, 4))

        self.nodes = [
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
        ]

        self.graph_spec = GraphSpec(roi=Roi((-100, -100, -100), (300, 300, 300)))
        self.array_spec = ArraySpec(
            roi=Roi((-200, -200, -200), (400, 400, 400)), voxel_size=self.voxel_size
        )

        self.graph = Graph(self.nodes, [], self.graph_spec)

    def setup(self):
        self.provides(
            GraphKeys.TEST_GRAPH,
            self.graph_spec,
        )

        self.provides(
            ArrayKeys.GT_LABELS,
            self.array_spec,
        )

    def provide(self, request):
        batch = Batch()

        graph_roi = request[GraphKeys.TEST_GRAPH].roi

        batch.graphs[GraphKeys.TEST_GRAPH] = self.graph.crop(graph_roi).trim(graph_roi)

        roi_array = request[ArrayKeys.GT_LABELS].roi

        image = np.ones(roi_array.shape / self.voxel_size, dtype=np.uint64)
        # label half of GT_LABELS differently
        depth = image.shape[0]
        image[0 : depth // 2] = 2

        spec = self.spec[ArrayKeys.GT_LABELS].copy()
        spec.roi = roi_array
        batch.arrays[ArrayKeys.GT_LABELS] = Array(image, spec=spec)

        return batch


class GraphTestSourceWithEdge(BatchProvider):
    def __init__(self):
        self.voxel_size = Coordinate((1, 1, 1))

        self.nodes = [
            # corners
            Node(id=1, location=np.array((0, 4, 4))),
            Node(id=2, location=np.array((9, 4, 4))),
        ]
        self.edges = [Edge(1, 2)]

        self.graph_spec = GraphSpec(roi=Roi((0, 0, 0), (10, 10, 10)))
        self.graph = Graph(self.nodes, self.edges, self.graph_spec)

    def setup(self):
        self.provides(
            GraphKeys.TEST_GRAPH_WITH_EDGE,
            self.graph_spec,
        )

    def provide(self, request):
        batch = Batch()

        graph_roi = request[GraphKeys.TEST_GRAPH_WITH_EDGE].roi

        batch.graphs[GraphKeys.TEST_GRAPH_WITH_EDGE] = self.graph.crop(graph_roi).trim(
            graph_roi
        )

        return batch


class TestRasterizePoints(ProviderTest):
    def test_3d(self):
        GraphKey("TEST_GRAPH")
        ArrayKey("RASTERIZED")

        pipeline = GraphTestSource3D() + RasterizeGraph(
            GraphKeys.TEST_GRAPH,
            ArrayKeys.RASTERIZED,
            ArraySpec(voxel_size=(40, 4, 4)),
        )

        with build(pipeline):
            request = BatchRequest()
            roi = Roi((0, 0, 0), (200, 200, 200))

            request[GraphKeys.TEST_GRAPH] = GraphSpec(roi=roi)
            request[ArrayKeys.GT_LABELS] = ArraySpec(roi=roi)
            request[ArrayKeys.RASTERIZED] = ArraySpec(roi=roi)

            batch = pipeline.request_batch(request)

            rasterized = batch.arrays[ArrayKeys.RASTERIZED].data
            self.assertEqual(rasterized[0, 0, 0], 1)
            self.assertEqual(rasterized[2, 20, 20], 0)
            self.assertEqual(rasterized[4, 49, 49], 1)

        # same with different foreground/background labels

        pipeline = GraphTestSource3D() + RasterizeGraph(
            GraphKeys.TEST_GRAPH,
            ArrayKeys.RASTERIZED,
            ArraySpec(voxel_size=(40, 4, 4)),
            RasterizationSettings(radius=1, fg_value=0, bg_value=1),
        )

        with build(pipeline):
            request = BatchRequest()
            roi = Roi((0, 0, 0), (200, 200, 200))

            request[GraphKeys.TEST_GRAPH] = GraphSpec(roi=roi)
            request[ArrayKeys.GT_LABELS] = ArraySpec(roi=roi)
            request[ArrayKeys.RASTERIZED] = ArraySpec(roi=roi)

            batch = pipeline.request_batch(request)

            rasterized = batch.arrays[ArrayKeys.RASTERIZED].data
            self.assertEqual(rasterized[0, 0, 0], 0)
            self.assertEqual(rasterized[2, 20, 20], 1)
            self.assertEqual(rasterized[4, 49, 49], 0)

        # same with different radius and inner radius

        pipeline = GraphTestSource3D() + RasterizeGraph(
            GraphKeys.TEST_GRAPH,
            ArrayKeys.RASTERIZED,
            ArraySpec(voxel_size=(40, 4, 4)),
            RasterizationSettings(
                radius=40, inner_radius_fraction=0.25, fg_value=1, bg_value=0
            ),
        )

        with build(pipeline):
            request = BatchRequest()
            roi = Roi((0, 0, 0), (200, 200, 200))

            request[GraphKeys.TEST_GRAPH] = GraphSpec(roi=roi)
            request[ArrayKeys.GT_LABELS] = ArraySpec(roi=roi)
            request[ArrayKeys.RASTERIZED] = ArraySpec(roi=roi)

            batch = pipeline.request_batch(request)

            rasterized = batch.arrays[ArrayKeys.RASTERIZED].data

            # in the middle of the ball, there should be 0 (since inner radius is set)
            self.assertEqual(rasterized[0, 0, 0], 0)
            # check larger radius: rasterized point (0, 0, 0) should extend in
            # x,y by 10; z, by 1
            self.assertEqual(rasterized[0, 10, 0], 1)
            self.assertEqual(rasterized[0, 0, 10], 1)
            self.assertEqual(rasterized[1, 0, 0], 1)

            self.assertEqual(rasterized[2, 20, 20], 0)
            self.assertEqual(rasterized[4, 49, 49], 0)

        # same with anisotropic radius

        pipeline = GraphTestSource3D() + RasterizeGraph(
            GraphKeys.TEST_GRAPH,
            ArrayKeys.RASTERIZED,
            ArraySpec(voxel_size=(40, 4, 4)),
            RasterizationSettings(radius=(40, 40, 20), fg_value=1, bg_value=0),
        )

        with build(pipeline):
            request = BatchRequest()
            roi = Roi((0, 0, 0), (120, 80, 80))

            request[GraphKeys.TEST_GRAPH] = GraphSpec(roi=roi)
            request[ArrayKeys.GT_LABELS] = ArraySpec(roi=roi)
            request[ArrayKeys.RASTERIZED] = ArraySpec(roi=roi)

            batch = pipeline.request_batch(request)

            rasterized = batch.arrays[ArrayKeys.RASTERIZED].data

            # check larger radius: rasterized point (0, 0, 0) should extend in
            # x,y by 10; z, by 1
            self.assertEqual(rasterized[0, 10, 0], 1)
            self.assertEqual(rasterized[0, 11, 0], 0)
            self.assertEqual(rasterized[0, 0, 5], 1)
            self.assertEqual(rasterized[0, 0, 6], 0)
            self.assertEqual(rasterized[1, 0, 0], 1)
            self.assertEqual(rasterized[2, 0, 0], 0)

        # same with anisotropic radius and inner radius

        pipeline = GraphTestSource3D() + RasterizeGraph(
            GraphKeys.TEST_GRAPH,
            ArrayKeys.RASTERIZED,
            ArraySpec(voxel_size=(40, 4, 4)),
            RasterizationSettings(
                radius=(40, 40, 20), inner_radius_fraction=0.75, fg_value=1, bg_value=0
            ),
        )

        with build(pipeline):
            request = BatchRequest()
            roi = Roi((0, 0, 0), (120, 80, 80))

            request[GraphKeys.TEST_GRAPH] = GraphSpec(roi=roi)
            request[ArrayKeys.GT_LABELS] = ArraySpec(roi=roi)
            request[ArrayKeys.RASTERIZED] = ArraySpec(roi=roi)

            batch = pipeline.request_batch(request)

            rasterized = batch.arrays[ArrayKeys.RASTERIZED].data

            # in the middle of the ball, there should be 0 (since inner radius is set)
            self.assertEqual(rasterized[0, 0, 0], 0)
            # check larger radius: rasterized point (0, 0, 0) should extend in
            # x,y by 10; z, by 1
            self.assertEqual(rasterized[0, 10, 0], 1)
            self.assertEqual(rasterized[0, 11, 0], 0)
            self.assertEqual(rasterized[0, 0, 5], 1)
            self.assertEqual(rasterized[0, 0, 6], 0)
            self.assertEqual(rasterized[1, 0, 0], 1)
            self.assertEqual(rasterized[2, 0, 0], 0)

    def test_with_edge(self):
        graph_with_edge = GraphKey("TEST_GRAPH_WITH_EDGE")
        array_with_edge = ArrayKey("RASTERIZED_EDGE")

        pipeline = GraphTestSourceWithEdge() + RasterizeGraph(
            GraphKeys.TEST_GRAPH_WITH_EDGE,
            ArrayKeys.RASTERIZED_EDGE,
            ArraySpec(voxel_size=(1, 1, 1)),
            settings=RasterizationSettings(0.5),
        )

        with build(pipeline):
            request = BatchRequest()
            roi = Roi((0, 0, 0), (10, 10, 10))

            request[GraphKeys.TEST_GRAPH_WITH_EDGE] = GraphSpec(roi=roi)
            request[ArrayKeys.RASTERIZED_EDGE] = ArraySpec(roi=roi)

            batch = pipeline.request_batch(request)

            rasterized = batch.arrays[ArrayKeys.RASTERIZED_EDGE].data

            assert (
                rasterized.sum() == 10
            ), f"rasterized has ones at: {np.where(rasterized==1)}"
