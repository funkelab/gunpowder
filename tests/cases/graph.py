import numpy as np

from gunpowder import (
    Batch,
    BatchFilter,
    BatchProvider,
    BatchRequest,
    Coordinate,
    Edge,
    Graph,
    GraphKey,
    GraphSpec,
    Node,
    Roi,
    build,
)


class ExampleGraphSource(BatchProvider):
    def __init__(self, graph_key):
        self.graph_key = graph_key
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
        self.provides(self.graph_key, self.__spec)

    def provide(self, request):
        batch = Batch()

        roi = request[self.graph_key].roi

        sub_graph = self.graph.crop(roi)

        batch[self.graph_key] = sub_graph

        return batch


class GrowFilter(BatchFilter):
    def __init__(self, graph_key):
        self.graph_key = graph_key

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


def edges():
    return [Edge(0, 1), Edge(1, 2), Edge(2, 3), Edge(3, 4), Edge(4, 0)]


def nodes():
    return [
        Node(0, location=np.array([0, 0, 0])),
        Node(1, location=np.array([1, 1, 1])),
        Node(2, location=np.array([2, 2, 2])),
        Node(3, location=np.array([3, 3, 3])),
        Node(4, location=np.array([4, 4, 4])),
    ]


def spec():
    return GraphSpec(
        roi=Roi(Coordinate([0, 0, 0]), Coordinate([5, 5, 5])), directed=True
    )


def test_output():
    graph_key = GraphKey("TEST_GRAPH")

    pipeline = ExampleGraphSource(graph_key) + GrowFilter(graph_key)

    with build(pipeline):
        batch = pipeline.request_batch(
            BatchRequest({graph_key: GraphSpec(roi=Roi((0, 0, 0), (50, 50, 50)))})
        )

        graph = batch[graph_key]
        expected_vertices = (
            Node(id=1, location=np.array([1.0, 1.0, 1.0], dtype=float)),
            Node(
                id=2,
                location=np.array([50.0, 50.0, 50.0], dtype=float),
                temporary=True,
            ),
        )
        seen_vertices = tuple(graph.nodes)
        assert sorted(
            [
                v.original_id if v.original_id is not None else -1
                for v in expected_vertices
            ]
        ) == sorted(
            [v.original_id if v.original_id is not None else -1 for v in seen_vertices]
        )
        for expected, actual in zip(
            sorted(expected_vertices, key=lambda v: tuple(v.location)),
            sorted(seen_vertices, key=lambda v: tuple(v.location)),
        ):
            assert all(np.isclose(expected.location, actual.location))

        batch = pipeline.request_batch(
            BatchRequest({graph_key: GraphSpec(roi=Roi((25, 25, 25), (500, 500, 500)))})
        )

        graph = batch[graph_key]
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
        assert sorted(
            [
                v.original_id if v.original_id is not None else -1
                for v in expected_vertices
            ]
        ) == sorted(
            [v.original_id if v.original_id is not None else -1 for v in seen_vertices]
        )
        for expected, actual in zip(
            sorted(expected_vertices, key=lambda v: tuple(v.location)),
            sorted(seen_vertices, key=lambda v: tuple(v.location)),
        ):
            assert all(np.isclose(expected.location, actual.location))


def test_neighbors():
    # directed
    d_spec = spec()
    # undirected
    ud_spec = spec()
    ud_spec.directed = False

    directed = Graph(nodes(), edges(), d_spec)
    undirected = Graph(nodes(), edges(), ud_spec)

    assert [x for x in directed.neighbors(nodes()[0])] == [
        x for x in undirected.neighbors(nodes()[0])
    ]


def test_crop():
    g = Graph(nodes(), edges(), spec())

    sub_g = g.crop(Roi(Coordinate([1, 1, 1]), Coordinate([3, 3, 3])))
    assert g.spec.roi == spec().roi
    assert sub_g.spec.roi == Roi(Coordinate([1, 1, 1]), Coordinate([3, 3, 3]))

    sub_g.spec.directed = False
    assert g.spec.directed
    assert not sub_g.spec.directed


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
