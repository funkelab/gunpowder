from .provider_test import ProviderTest
import unittest
from gunpowder import (
    BatchProvider,
    BatchRequest,
    BatchFilter,
    Batch,
    Vertex,
    Edge,
    Graph,
    GraphSpec,
    GraphKey,
    GraphKeys,
    build,
    Roi,
    Coordinate,
)

import numpy as np


class TestGraphSource(BatchProvider):
    def __init__(self):

        self.dtype = int
        self.vertices = [
            Vertex(id=1, location=np.ndarray([1, 1, 1], dtype=self.dtype)),
            Vertex(id=2, location=np.ndarray([500, 500, 500], dtype=self.dtype)),
            Vertex(id=3, location=np.ndarray([550, 550, 550], dtype=self.dtype)),
        ]
        self.edges = [Edge(1, 2), Edge(2, 3)]
        self.spec = GraphSpec(
            roi=Roi(Coordinate([-500, -500, -500]), Coordinate([1500, 1500, 1500]))
        )
        self.graph = Graph(self.vertices, self.edges, self.spec)

    def setup(self):

        self.provides(GraphKeys.TEST_GRAPH, self.spec)

    def provide(self, request):

        batch = Batch()

        roi = request[GraphKeys.TEST_GRAPH].roi

        sub_graph = self.graph.crop(roi)

        batch[GraphKeys.TEST_GRAPH] = sub_graph

        return batch


class GrowFilter(BatchFilter):
    def prepare(self, request):
        grow = Coordinate([50, 50, 50])
        for key, spec in request.items():
            spec.roi = spec.roi.grow(grow, grow)
            request[key] = spec
        return request

    def process(self, batch, request):
        for key, spec in request.items():
            batch[key] = batch[key].crop(spec.roi).trim(spec.roi)
        return batch


class TestGraphs(ProviderTest):
    def test_output(self):
        """
        Request a roi, which gets grown, and includes a point not
        in the original request. Cropping should lead to new
        nodes begin placed at the intersection between the
        requested roi, and the edge crossing the bounding box.
        """

        GraphKey("TEST_GRAPH")

        pipeline = TestGraphSource() + GrowFilter()

        with build(pipeline):

            batch = pipeline.request_batch(
                BatchRequest(
                    {
                        GraphKeys.TEST_GRAPH: GraphSpec(
                            roi=Roi((525, 525, 525), (50, 50, 50))
                        )
                    }
                )
            )

            graph = batch[GraphKeys.TEST_GRAPH]
            expected_vertices = (
                Vertex(id=None, location=np.array([525, 525, 525], dtype=int)),
                Vertex(id=3, location=np.array([550, 550, 550], dtype=int)),
            )
            seen_vertices = tuple(graph.vertices)
            self.assertCountEqual(
                [v.id for v in expected_vertices], [v.id for v in seen_vertices]
            )
            self.assertCountEqual(
                [v.location for v in expected_vertices],
                [v.location for v in seen_vertices],
            )

            batch = pipeline.request_batch(
                BatchRequest(
                    {
                        GraphKeys.TEST_GRAPH: GraphSpec(
                            roi=Roi((25, 25, 25), (500, 500, 500))
                        )
                    }
                )
            )

            graph = batch[GraphKeys.TEST_GRAPH]
            expected_vertices = (
                Vertex(id=None, location=np.array([25, 25, 25], dtype=int)),
                Vertex(id=2, location=np.array([500, 500, 500], dtype=int)),
                Vertex(id=None, location=np.array([524, 524, 524], dtype=int)),
            )
            seen_vertices = tuple(graph.vertices)
            self.assertCountEqual(
                [v.id for v in expected_vertices], [v.id for v in seen_vertices]
            )
            self.assertCountEqual(
                [v.location for v in expected_vertices],
                [v.location for v in seen_vertices],
            )
