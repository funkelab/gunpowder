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


class ExampleGraphSource(BatchProvider):
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
    @property
    def edges(self):
        return [Edge(0, 1), Edge(1, 2), Edge(2, 3), Edge(3, 4), Edge(4, 0)]

    @property
    def nodes(self):
        return [
            Node(0, location=np.array([0, 0, 0], dtype=self.spec.dtype)),
            Node(1, location=np.array([1, 1, 1], dtype=self.spec.dtype)),
            Node(2, location=np.array([2, 2, 2], dtype=self.spec.dtype)),
            Node(3, location=np.array([3, 3, 3], dtype=self.spec.dtype)),
            Node(4, location=np.array([4, 4, 4], dtype=self.spec.dtype)),
        ]

    @property
    def spec(self):
        return GraphSpec(
            roi=Roi(Coordinate([0, 0, 0]), Coordinate([5, 5, 5])), directed=True
        )

    def test_output(self):
        GraphKey("TEST_GRAPH")

        pipeline = ExampleGraphSource() + GrowFilter()

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
        d_spec = self.spec
        # undirected
        ud_spec = self.spec
        ud_spec.directed = False

        directed = Graph(self.nodes, self.edges, d_spec)
        undirected = Graph(self.nodes, self.edges, ud_spec)

        self.assertCountEqual(
            directed.neighbors(self.nodes[0]), undirected.neighbors(self.nodes[0])
        )

    def test_crop(self):
        g = Graph(self.nodes, self.edges, self.spec)

        sub_g = g.crop(Roi(Coordinate([1, 1, 1]), Coordinate([3, 3, 3])))
        self.assertEqual(g.spec.roi, self.spec.roi)
        self.assertEqual(
            sub_g.spec.roi, Roi(Coordinate([1, 1, 1]), Coordinate([3, 3, 3]))
        )

        sub_g.spec.directed = False
        self.assertTrue(g.spec.directed)
        self.assertFalse(sub_g.spec.directed)


def test_nodes():
    initial_locations = {
        1: np.array([1, 1, 1], dtype=np.float32),
        2: np.array([500, 500, 500], dtype=np.float32),
        3: np.array([550, 550, 550], dtype=np.float32),
    }
    replacement_locations = {
        1: np.array([0, 0, 0], dtype=np.float32),
        2: np.array([50, 50, 50], dtype=np.float32),
        3: np.array([55, 55, 55], dtype=np.float32),
    }

    nodes = [
        Node(id=id, location=location) for id, location in initial_locations.items()
    ]
    edges = [Edge(1, 2), Edge(2, 3)]
    spec = GraphSpec(
        roi=Roi(Coordinate([-500, -500, -500]), Coordinate([1500, 1500, 1500]))
    )
    graph = Graph(nodes, edges, spec)
    for node in graph.nodes:
        node.location = replacement_locations[node.id]

    for node in graph.nodes:
        assert all(np.isclose(node.location, replacement_locations[node.id]))
