import unittest
from gunpowder import (
    GraphSpec,
    GraphKey,
    Roi,
    Coordinate,
    Batch,
    BatchProvider,
    ArrayKeys,
    ArrayKey,
    ArraySpec,
    Array,
    BatchRequest,
    MergeProvider,
    RandomLocation,
    build,
)
from gunpowder.graph import GraphKeys, Graph
from gunpowder.pipeline import PipelineSetupError

import numpy as np


class GraphTestSource(BatchProvider):
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def setup(self):
        self.provides(GraphKeys.PRESYN, GraphSpec(roi=Roi((0, 0, 0), (100, 100, 100))))

    def provide(self, request):
        batch = Batch()
        graph_roi = request[GraphKeys.PRESYN].roi

        batch.graphs[GraphKeys.PRESYN] = Graph([], [], GraphSpec(roi=graph_roi))
        return batch


class ArrayTestSoure(BatchProvider):
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def setup(self):
        self.provides(
            ArrayKeys.GT_LABELS,
            ArraySpec(roi=Roi((0, 0, 0), (100, 100, 100)), voxel_size=self.voxel_size),
        )

    def provide(self, request):
        roi_array = request[ArrayKeys.GT_LABELS].roi
        data = np.zeros(roi_array.shape / self.spec[ArrayKeys.GT_LABELS].voxel_size)
        batch = Batch()
        spec = self.spec[ArrayKeys.GT_LABELS].copy()
        spec.roi = roi_array
        batch.arrays[ArrayKeys.GT_LABELS] = Array(data, spec)
        return batch


class TestMergeProvider(unittest.TestCase):
    def test_merge_basics(self):
        voxel_size = (1, 1, 1)
        GraphKey("PRESYN")
        ArrayKey("GT_LABELS")
        graphsource = GraphTestSource(voxel_size)
        arraysource = ArrayTestSoure(voxel_size)
        pipeline = (graphsource, arraysource) + MergeProvider() + RandomLocation()
        window_request = Coordinate((50, 50, 50))
        with build(pipeline):
            # Check basic merging.
            request = BatchRequest()
            request.add((GraphKeys.PRESYN), window_request)
            request.add((ArrayKeys.GT_LABELS), window_request)
            batch_res = pipeline.request_batch(request)
            self.assertTrue(ArrayKeys.GT_LABELS in batch_res.arrays)
            self.assertTrue(GraphKeys.PRESYN in batch_res.graphs)

            # Check that request of only one source also works.
            request = BatchRequest()
            request.add((GraphKeys.PRESYN), window_request)
            batch_res = pipeline.request_batch(request)
            self.assertFalse(ArrayKeys.GT_LABELS in batch_res.arrays)
            self.assertTrue(GraphKeys.PRESYN in batch_res.graphs)

        # Check that it fails, when having two sources that provide the same type.
        arraysource2 = ArrayTestSoure(voxel_size)
        pipeline_fail = (arraysource, arraysource2) + MergeProvider() + RandomLocation()
        with self.assertRaises(PipelineSetupError):
            with build(pipeline_fail):
                pass
