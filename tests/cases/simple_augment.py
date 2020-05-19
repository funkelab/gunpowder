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
            [Node(id=1, location=np.array([1, 1, 1]))],
            [],
            GraphSpec(roi=Roi((0, 0, 0), (100, 100, 100))),
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
            mirror_only=[0, 1, 2], transpose_only=[]
        )

        request = BatchRequest()
        request[GraphKeys.TEST_GRAPH] = GraphSpec(roi=Roi((0, 0, 0), (100, 100, 100)))

        with build(pipeline):
            seen_mirrored = False
            for i in range(100):
                batch = pipeline.request_batch(request)

                assert len(list(batch[GraphKeys.TEST_GRAPH].nodes)) == 1
                node = list(batch[GraphKeys.TEST_GRAPH].nodes)[0]
                assert all(
                    [
                        node.location[dim] == 1 or node.location[dim] == 99
                        for dim in range(3)
                    ]
                )
                seen_mirrored = seen_mirrored or any(
                    [node.location[dim] == 99 for dim in range(3)]
                )
            assert seen_mirrored
