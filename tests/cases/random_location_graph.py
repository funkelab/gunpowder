from .provider_test import ProviderTest
from gunpowder import (
    BatchProvider,
    Graph,
    Vertex,
    GraphSpec,
    GraphKey,
    GraphKeys,
    Roi,
    Batch,
    BatchRequest,
    RandomLocation,
    build,
)
import numpy as np


class TestSourceRandomLocation(BatchProvider):
    def __init__(self):

        self.graph = Graph(
            [
                Vertex(id=1, location=np.array([1, 1, 1])),
                Vertex(id=2, location=np.array([500, 500, 500])),
                Vertex(id=3, location=np.array([550, 550, 550])),
            ],
            [],
            GraphSpec(roi=Roi((0, 0, 0), (1000, 1000, 1000))),
        )

    def setup(self):

        self.provides(GraphKeys.TEST_GRAPH, self.graph.spec)

    def provide(self, request):

        batch = Batch()

        roi = request[GraphKeys.TEST_GRAPH].roi
        batch[GraphKeys.TEST_GRAPH] = self.graph.crop(roi, copy=True).trim(roi)

        return batch


class TestRandomLocationGraph(ProviderTest):
    def test_output(self):

        GraphKey("TEST_GRAPH")

        pipeline = TestSourceRandomLocation() + RandomLocation(
            ensure_nonempty=GraphKeys.TEST_GRAPH
        )

        # count the number of times we get each vertex
        histogram = {}

        with build(pipeline):

            for i in range(5000):
                batch = pipeline.request_batch(
                    BatchRequest(
                        {
                            GraphKeys.TEST_GRAPH: GraphSpec(
                                roi=Roi((0, 0, 0), (100, 100, 100))
                            )
                        }
                    )
                )

                vertices = list(batch[GraphKeys.TEST_GRAPH].vertices)
                vertex_ids = [v.id for v in vertices]

                self.assertTrue(len(vertices) > 0)
                self.assertTrue(
                    (1 in vertex_ids) != (2 in vertex_ids or 3 in vertex_ids),
                    vertex_ids,
                )

                for vertex in batch[GraphKeys.TEST_GRAPH].vertices:
                    if vertex.id not in histogram:
                        histogram[vertex.id] = 1
                    else:
                        histogram[vertex.id] += 1

        total = sum(histogram.values())
        for k, v in histogram.items():
            histogram[k] = float(v) / total

        # we should get roughly the same count for each point
        for i in histogram.keys():
            for j in histogram.keys():
                self.assertAlmostEqual(histogram[i], histogram[j], 1)
