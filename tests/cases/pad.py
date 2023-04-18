from .provider_test import ProviderTest
from gunpowder import (
    BatchProvider,
    BatchRequest,
    Batch,
    ArrayKeys,
    ArraySpec,
    Roi,
    Coordinate,
    GraphKey,
    GraphKeys,
    GraphSpec,
    Array,
    ArrayKey,
    Pad,
    build,
)
import numpy as np


class ExampleSourcePad(BatchProvider):
    def setup(self):
        self.provides(
            ArrayKeys.TEST_LABELS,
            ArraySpec(roi=Roi((200, 20, 20), (1800, 180, 180)), voxel_size=(20, 2, 2)),
        )

        self.provides(
            GraphKeys.TEST_GRAPH, GraphSpec(roi=Roi((200, 20, 20), (1800, 180, 180)))
        )

    def provide(self, request):
        batch = Batch()

        roi_array = request[ArrayKeys.TEST_LABELS].roi
        roi_voxel = roi_array // self.spec[ArrayKeys.TEST_LABELS].voxel_size

        data = np.zeros(roi_voxel.shape, dtype=np.uint32)
        data[:, ::2] = 100

        spec = self.spec[ArrayKeys.TEST_LABELS].copy()
        spec.roi = roi_array
        batch.arrays[ArrayKeys.TEST_LABELS] = Array(data, spec=spec)

        return batch


class TestPad(ProviderTest):
    def test_output(self):
        graph = GraphKey("TEST_GRAPH")
        labels = ArrayKey("TEST_LABELS")

        pipeline = (
            ExampleSourcePad()
            + Pad(labels, Coordinate((20, 20, 20)), value=1)
            + Pad(graph, Coordinate((10, 10, 10)))
        )

        with build(pipeline):
            self.assertTrue(
                pipeline.spec[labels].roi == Roi((180, 0, 0), (1840, 220, 220))
            )
            self.assertTrue(
                pipeline.spec[graph].roi == Roi((190, 10, 10), (1820, 200, 200))
            )

            batch = pipeline.request_batch(
                BatchRequest({labels: ArraySpec(Roi((180, 0, 0), (20, 20, 20)))})
            )

            self.assertEqual(np.sum(batch.arrays[labels].data), 1 * 10 * 10)
