import logging

import numpy as np

from gunpowder import (
    Batch,
    BatchFilter,
    BatchProvider,
    BatchRequest,
    Graph,
    GraphKey,
    GraphKeys,
    GraphSpec,
    Node,
    RandomLocation,
    Roi,
    build,
)

logger = logging.getLogger(__name__)


class BatchTester(BatchFilter):
    def __init__(self, roi_to_match, exact=True):
        self.roi_to_match = roi_to_match
        self.exact = exact
        self.visted = False

    def prepare(self, request):
        for key, v in request.items():
            logger.debug(f"PREPARE TESTBATCH ======== {key} ROI: {self.spec[key].roi}")

    def process(self, batch, request):
        if self.visted:
            for key, graph in batch.graphs.items():
                logger.debug(
                    f"PROCESS TESTBATCH ======== {key}: {graph.spec.roi} {graph}"
                )
                if self.exact:
                    assert (
                        graph.spec.roi == self.roi_to_match
                    ), "graph roi does not match possible roi"
                else:
                    assert self.roi_to_match.contains(
                        batch[GraphKeys.TEST_GRAPH].spec.roi
                    ), "batch is not contained in possible roi"
        else:
            self.visted = True


class SourceGraphLocation(BatchProvider):
    def __init__(self, graph_key):
        self.graph_key = graph_key
        self.graph = Graph(
            [Node(id=1, location=np.array([500, 500, 500]))],
            [],
            GraphSpec(roi=Roi((0, 0, 0), (1000, 1000, 1000))),
        )

    def setup(self):
        self.provides(self.graph_key, self.graph.spec)

    def provide(self, request):
        batch = Batch()

        roi = request[self.graph_key].roi
        batch[self.graph_key] = self.graph.crop(roi).trim(roi)

        return batch


def test_dim_size_1():
    graph_key = GraphKey("TEST_GRAPH")
    upstream_roi = Roi((500, 401, 401), (1, 200, 200))
    pipeline = (
        SourceGraphLocation(graph_key)
        + BatchTester(upstream_roi, exact=False)
        + RandomLocation(ensure_nonempty=graph_key)
    )

    # count the number of times we get each node
    with build(pipeline):
        for _ in range(50):
            batch = pipeline.request_batch(
                BatchRequest({graph_key: GraphSpec(roi=Roi((0, 0, 0), (1, 100, 100)))})
            )

            assert len(list(batch[graph_key].nodes)) == 1


def test_req_full_roi():
    graph_key = GraphKey("TEST_GRAPH")

    possible_roi = Roi((0, 0, 0), (1000, 1000, 1000))

    pipeline = (
        SourceGraphLocation(graph_key)
        + BatchTester(possible_roi, exact=False)
        + RandomLocation(ensure_nonempty=graph_key)
    )
    with build(pipeline):
        batch = pipeline.request_batch(
            BatchRequest({graph_key: GraphSpec(roi=Roi((0, 0, 0), (1000, 1000, 1000)))})
        )

        assert len(list(batch[graph_key].nodes)) == 1


def test_roi_one_point():
    graph_key = GraphKey("TEST_GRAPH")
    upstream_roi = Roi((500, 500, 500), (1, 1, 1))

    pipeline = (
        SourceGraphLocation(graph_key)
        + BatchTester(upstream_roi, exact=True)
        + RandomLocation(ensure_nonempty=graph_key)
    )

    with build(pipeline):
        for _ in range(50):
            batch = pipeline.request_batch(
                BatchRequest({graph_key: GraphSpec(roi=Roi((0, 0, 0), (1, 1, 1)))})
            )

            assert len(list(batch[graph_key].nodes)) == 1


def test_iso_roi():
    graph_key = GraphKey("TEST_GRAPH")
    upstream_roi = Roi((401, 401, 401), (200, 200, 200))

    pipeline = (
        SourceGraphLocation(graph_key)
        + BatchTester(upstream_roi, exact=False)
        + RandomLocation(ensure_nonempty=graph_key)
    )

    with build(pipeline):
        for _ in range(50):
            batch = pipeline.request_batch(
                BatchRequest(
                    {graph_key: GraphSpec(roi=Roi((0, 0, 0), (100, 100, 100)))}
                )
            )

            assert len(list(batch[graph_key].nodes)) == 1
