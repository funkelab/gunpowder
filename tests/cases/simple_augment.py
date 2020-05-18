from .provider_test import ProviderTest
from gunpowder import (
    IntensityAugment,
    ArrayKeys,
    build,
    Normalize,
    Graph,
    Node,
    GraphSpec,
    Roi,
    BatchProvider,
    BatchRequest,
    GraphKeys,
    GraphKey,
    Batch,
    SimpleAugment,
)

import numpy as np


class TestSource(BatchProvider):
    def __init__(self):

        self.graph = Graph(
            [
                Node(id=1, location=np.array([1, 1, 1])),
                Node(id=2, location=np.array([450, 450, 450])),
                Node(id=3, location=np.array([551, 551, 551])),
            ],
            [],
            GraphSpec(roi=Roi((0, 0, 0), (1000, 1000, 1000))),
        )

    def setup(self):

        self.provides(GraphKeys.TEST_GRAPH, self.graph.spec)

    def prepare(self, request):
        return request

    def provide(self, request):

        batch = Batch()

        roi = request[GraphKeys.TEST_GRAPH].roi
        batch[GraphKeys.TEST_GRAPH] = self.graph.crop(roi, copy=True).trim(roi)

        return batch


class TestSimpleAugment(ProviderTest):
    def test_simple(self):
        test_graph = GraphKey("TEST_GRAPH")

        pipeline = TestSource() + SimpleAugment(
            mirror_only=[0, 1, 2], transpose_only=[0, 1, 2]
        )

        request = BatchRequest()
        request[GraphKeys.TEST_GRAPH] = GraphSpec(
            roi=Roi((445, 445, 445), (10, 10, 10))
        )

        with build(pipeline):
            for i in range(100):
                batch = pipeline.request_batch(request)

                assert len(list(batch[GraphKeys.TEST_GRAPH].nodes)) > 0
