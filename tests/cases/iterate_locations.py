import networkx as nx
import numpy as np

from gunpowder import (
    ArrayKey,
    ArraySpec,
    BatchRequest,
    Coordinate,
    Edge,
    GraphKey,
    GraphSource,
    GraphSpec,
    IterateLocations,
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
        for node in self.nodes:
            if roi.contains(node.location):
                graph.add_node(node.id, location=node.location)
        for edge in self.edges:
            if edge.u in graph.nodes:
                graph.add_edge(edge.u, edge.v)
        return graph


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
    node_key = ArrayKey("NODE_ID")

    dummy_provider = DummyDaisyGraphProvider(nodes(), edges(), directed=True)
    graph_source = GraphSource(dummy_provider, graph_key, spec())
    iterate_locations = IterateLocations(graph_key, node_id=node_key)
    pipeline = graph_source + iterate_locations
    request = BatchRequest(
        {
            graph_key: GraphSpec(roi=Roi((0, 0, 0), (1, 1, 1))),
            node_key: ArraySpec(nonspatial=True),
        }
    )
    node_ids = []
    seen_vertices = []
    expected_vertices = nodes()
    with build(pipeline):
        for _ in range(len(nodes())):
            batch = pipeline.request_batch(request)
            node_ids.extend(batch[node_key].data)
            graph = batch[graph_key]
            assert graph.num_vertices() == 1
            node = next(graph.nodes)
            seen_vertices.append(node)

        assert [v.id for v in expected_vertices] == node_ids
        for vertex in seen_vertices:
            # locations are shifted to lie in roi (so, (0, 0, 0))
            assert all(np.isclose(np.array([0.0, 0.0, 0.0]), vertex.location))
