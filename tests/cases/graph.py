from .provider_test import ProviderTest
from gunpowder import (
    BatchProvider,
    BatchRequest,
    BatchFilter,
    Batch,
    Node,
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

        self.dtype = float
        self.__vertices = [
            Node(id=1, location=np.array([1, 1, 1], dtype=self.dtype)),
            Node(id=2, location=np.array([500, 500, 500], dtype=self.dtype)),
            Node(id=3, location=np.array([550, 550, 550], dtype=self.dtype)),
        ]
        self.__edges = [Edge(1, 2), Edge(2, 3)]
        self.__spec = GraphSpec(
            roi=Roi(Coordinate([-500, -500, -500]), Coordinate([1500, 1500, 1500]))
        )
        self.graph = Graph(self.__vertices, self.__edges, self.__spec)

    def setup(self):

        self.provides(GraphKeys.TEST_GRAPH, self.__spec)

    def provide(self, request):

        batch = Batch()

        roi = request[GraphKeys.TEST_GRAPH].roi

        sub_graph = self.graph.crop(roi, copy=True)

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

        GraphKey("TEST_GRAPH")

        pipeline = TestGraphSource() + GrowFilter()

        with build(pipeline):

            batch = pipeline.request_batch(
                BatchRequest(
                    {GraphKeys.TEST_GRAPH: GraphSpec(roi=Roi((0, 0, 0), (50, 50, 50)))}
                )
            )

            graph = batch[GraphKeys.TEST_GRAPH]
            expected_vertices = (
                Node(id=1, location=np.array([1.0, 1.0, 1.0], dtype=float)),
                Node(
                    id=2,
                    location=np.array([50.0, 50.0, 50.0], dtype=float),
                    temporary=True,
                ),
            )
            seen_vertices = tuple(graph.nodes)
            self.assertCountEqual(
                [v.original_id for v in expected_vertices],
                [v.original_id for v in seen_vertices],
            )
            for expected, actual in zip(
                sorted(expected_vertices, key=lambda v: tuple(v.location)),
                sorted(seen_vertices, key=lambda v: tuple(v.location)),
            ):
                assert all(np.isclose(expected.location, actual.location))

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
                Node(
                    id=1,
                    location=np.array([25.0, 25.0, 25.0], dtype=float),
                    temporary=True,
                ),
                Node(id=2, location=np.array([500.0, 500.0, 500.0], dtype=float)),
                Node(
                    id=3,
                    location=np.array([525.0, 525.0, 525.0], dtype=float),
                    temporary=True,
                ),
            )
            seen_vertices = tuple(graph.nodes)
            self.assertCountEqual(
                [v.original_id for v in expected_vertices],
                [v.original_id for v in seen_vertices],
            )
            for expected, actual in zip(
                sorted(expected_vertices, key=lambda v: tuple(v.location)),
                sorted(seen_vertices, key=lambda v: tuple(v.location)),
            ):
                assert all(np.isclose(expected.location, actual.location))

    def test_neighbors(self):
        # directed
        d_spec = GraphSpec(
            roi=Roi(Coordinate([0, 0, 0]), Coordinate([5, 5, 5])), directed=True
        )
        # undirected
        ud_spec = GraphSpec(
            roi=Roi(Coordinate([0, 0, 0]), Coordinate([5, 5, 5])), directed=False
        )
        nodes = [
            Node(0, location=np.array([0, 0, 0], dtype=d_spec.dtype)),
            Node(1, location=np.array([1, 1, 1], dtype=d_spec.dtype)),
            Node(2, location=np.array([2, 2, 2], dtype=d_spec.dtype)),
            Node(3, location=np.array([3, 3, 3], dtype=d_spec.dtype)),
            Node(4, location=np.array([4, 4, 4], dtype=d_spec.dtype)),
        ]
        edges = [Edge(0, 1), Edge(1, 2), Edge(2, 3), Edge(3, 4), Edge(4, 0)]

        directed = Graph(nodes, edges, d_spec)
        undirected = Graph(nodes, edges, ud_spec)

        self.assertCountEqual(
            directed.neighbors(nodes[0]), undirected.neighbors(nodes[0])
        )
