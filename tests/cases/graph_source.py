import networkx as nx
import numpy as np

from gunpowder import (
    BatchRequest,
    Coordinate,
    Edge,
    GraphKey,
    GraphSource,
    GraphSpec,
    Node,
    Roi,
    build,
)


class DummyDaisyGraphProvider:
    """Dummy graph provider mimicing daisy.SharedGraphProvider.
    Must have directed attribute, __getitem__(roi) that returns networkx
    graph, and position_attribute.
    """

    def __init__(self, nodes, edges, directed=False):
        self.nodes = nodes
        self.edges = edges
        self.directed = directed
        self.position_attribute = "location"

    def __getitem__(self, roi):
        if self.directed:
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()
        for node in nodes():
            if roi.contains(node.location):
                graph.add_node(node.id, location=node.location)
        for edge in edges():
            if edge.u in graph.nodes:
                graph.add_edge(edge.u, edge.v)
        return graph


def edges():
    return [Edge(0, 1), Edge(1, 2), Edge(2, 3), Edge(3, 4), Edge(4, 0)]


def nodes():
    return [
        Node(0, location=np.array([0, 0, 0], dtype=spec().dtype)),
        Node(1, location=np.array([1, 1, 1], dtype=spec().dtype)),
        Node(2, location=np.array([2, 2, 2], dtype=spec().dtype)),
        Node(3, location=np.array([3, 3, 3], dtype=spec().dtype)),
        Node(4, location=np.array([4, 4, 4], dtype=spec().dtype)),
    ]


def spec():
    return GraphSpec(
        roi=Roi(Coordinate([0, 0, 0]), Coordinate([5, 5, 5])), directed=True
    )


def test_output():
    graph_key = GraphKey("TEST_GRAPH")

    dummy_provider = DummyDaisyGraphProvider(nodes(), edges(), directed=True)
    graph_source = GraphSource(dummy_provider, graph_key, spec())

    pipeline = graph_source

    with build(pipeline):
        batch = pipeline.request_batch(
            BatchRequest({graph_key: GraphSpec(roi=Roi((0, 0, 0), (5, 5, 5)))})
        )

        graph = batch[graph_key]
        expected_vertices = nodes()
        seen_vertices = tuple(graph.nodes)
        assert [v.id for v in expected_vertices] == [v.id for v in seen_vertices]
        for expected, actual in zip(
            sorted(expected_vertices, key=lambda v: tuple(v.location)),
            sorted(seen_vertices, key=lambda v: tuple(v.location)),
        ):
            assert all(np.isclose(expected.location, actual.location))

        batch = pipeline.request_batch(
            BatchRequest({graph_key: GraphSpec(roi=Roi((2, 2, 2), (3, 3, 3)))})
        )

        graph = batch[graph_key]
        expected_vertices = (
            Node(2, location=np.array([2, 2, 2], dtype=spec().dtype)),
            Node(3, location=np.array([3, 3, 3], dtype=spec().dtype)),
            Node(4, location=np.array([4, 4, 4], dtype=spec().dtype)),
        )
        seen_vertices = tuple(graph.nodes)
        print(seen_vertices)
        assert [v.id for v in expected_vertices] == [v.id for v in seen_vertices]
        for expected, actual in zip(
            sorted(expected_vertices, key=lambda v: tuple(v.location)),
            sorted(seen_vertices, key=lambda v: tuple(v.location)),
        ):
            assert all(np.isclose(expected.location, actual.location))
