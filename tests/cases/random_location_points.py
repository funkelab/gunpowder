from .provider_test import ProviderTest
from gunpowder import *
import numpy as np

class ExampleSourceRandomLocation(BatchProvider):

    def __init__(self):

        self.graph = Graph(
            [
                Node(1, np.array([1, 1, 1])),
                Node(2, np.array([500, 500, 500])),
                Node(3, np.array([550, 550, 550])),
            ],
            [],
            GraphSpec(
                roi=Roi((0, 0, 0), (1000, 1000, 1000))))

    def setup(self):

        self.provides(
            GraphKeys.TEST_POINTS,
            self.graph.spec)

    def provide(self, request):

        batch = Batch()

        roi = request[GraphKeys.TEST_POINTS].roi
        graph = Graph([], [], GraphSpec(roi))

        for node in self.graph.nodes:
            if roi.contains(node.location):
                graph.add_node(node)
        batch[GraphKeys.TEST_POINTS] = graph

        return batch

class TestRandomLocationPoints(ProviderTest):

    def test_output(self):

        GraphKey('TEST_POINTS')

        pipeline = (
            ExampleSourceRandomLocation() +
            RandomLocation(ensure_nonempty=GraphKeys.TEST_POINTS)
        )

        # count the number of times we get each point
        histogram = {}

        with build(pipeline):

            for i in range(5000):
                batch = pipeline.request_batch(
                    BatchRequest(
                        {
                            GraphKeys.TEST_POINTS: GraphSpec(
                                roi=Roi((0, 0, 0), (100, 100, 100)))
                        }))

                points = {node.id: node for node in batch[GraphKeys.TEST_POINTS].nodes}

                self.assertTrue(len(points) > 0)
                self.assertTrue((1 in points) != (2 in points or 3 in points), points)

                for node in batch[GraphKeys.TEST_POINTS].nodes:
                    if node.id not in histogram:
                        histogram[node.id] = 1
                    else:
                        histogram[node.id] += 1

        total = sum(histogram.values())
        for k, v in histogram.items():
            histogram[k] = float(v)/total

        # we should get roughly the same count for each point
        for i in histogram.keys():
            for j in histogram.keys():
                self.assertAlmostEqual(histogram[i], histogram[j], 1)
