from .provider_test import ProviderTest
from gunpowder import (
    BatchRequest,
    Node,
    Edge,
    GraphSpec,
    GraphKey,
    GraphKeys,
    GraphSource,
    build,
    Roi,
    Coordinate,
)

import numpy as np
import networkx as nx


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


class TestGraphSource(ProviderTest):
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

        dummy_provider = DummyDaisyGraphProvider(self.nodes, self.edges, directed=True)
        graph_source = GraphSource(dummy_provider, GraphKeys.TEST_GRAPH, self.spec)

        pipeline = graph_source

        with build(pipeline):
            batch = pipeline.request_batch(
                BatchRequest(
                    {GraphKeys.TEST_GRAPH: GraphSpec(roi=Roi((0, 0, 0), (5, 5, 5)))}
                )
            )

            graph = batch[GraphKeys.TEST_GRAPH]
            expected_vertices = self.nodes
            seen_vertices = tuple(graph.nodes)
            self.assertCountEqual(
                [v.id for v in expected_vertices],
                [v.id for v in seen_vertices],
            )
            for expected, actual in zip(
                sorted(expected_vertices, key=lambda v: tuple(v.location)),
                sorted(seen_vertices, key=lambda v: tuple(v.location)),
            ):
                assert all(np.isclose(expected.location, actual.location))

            batch = pipeline.request_batch(
                BatchRequest(
                    {GraphKeys.TEST_GRAPH: GraphSpec(roi=Roi((2, 2, 2), (3, 3, 3)))}
                )
            )

            graph = batch[GraphKeys.TEST_GRAPH]
            expected_vertices = (
                Node(2, location=np.array([2, 2, 2], dtype=self.spec.dtype)),
                Node(3, location=np.array([3, 3, 3], dtype=self.spec.dtype)),
                Node(4, location=np.array([4, 4, 4], dtype=self.spec.dtype)),
            )
            seen_vertices = tuple(graph.nodes)
            print(seen_vertices)
            self.assertCountEqual(
                [v.id for v in expected_vertices],
                [v.id for v in seen_vertices],
            )
            for expected, actual in zip(
                sorted(expected_vertices, key=lambda v: tuple(v.location)),
                sorted(seen_vertices, key=lambda v: tuple(v.location)),
            ):
                assert all(np.isclose(expected.location, actual.location))
