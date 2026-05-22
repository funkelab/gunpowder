import numpy as np
import pytest

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    Batch,
    BatchProvider,
    BatchRequest,
    Coordinate,
    GraphKey,
    GraphSpec,
    MergeProvider,
    RandomLocation,
    Roi,
    build,
)
from gunpowder.graph import Graph
from gunpowder.pipeline import PipelineSetupError

PRESYN = GraphKey("PRESYN")
GT_LABELS = ArrayKey("GT_LABELS")


class GraphTestSource(BatchProvider):
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def setup(self):
        self.provides(PRESYN, GraphSpec(roi=Roi((0, 0, 0), (100, 100, 100))))

    def provide(self, request):
        batch = Batch()
        graph_roi = request[PRESYN].roi

        batch.graphs[PRESYN] = Graph([], [], GraphSpec(roi=graph_roi))
        return batch


class ArrayTestSoure(BatchProvider):
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def setup(self):
        self.provides(
            GT_LABELS,
            ArraySpec(roi=Roi((0, 0, 0), (100, 100, 100)), voxel_size=self.voxel_size),
        )

    def provide(self, request):
        roi_array = request[GT_LABELS].roi
        data = np.zeros(roi_array.shape / self.spec[GT_LABELS].voxel_size)
        batch = Batch()
        spec = self.spec[GT_LABELS].copy()
        spec.roi = roi_array
        batch.arrays[GT_LABELS] = Array(data, spec)
        return batch


def test_merge_basics():
    voxel_size = (1, 1, 1)
    graphsource = GraphTestSource(voxel_size)
    arraysource = ArrayTestSoure(voxel_size)
    pipeline = (graphsource, arraysource) + MergeProvider() + RandomLocation()
    window_request = Coordinate((50, 50, 50))
    with build(pipeline):
        # Check basic merging.
        request = BatchRequest()
        request.add(PRESYN, window_request)
        request.add(GT_LABELS, window_request)
        batch_res = pipeline.request_batch(request)
        assert GT_LABELS in batch_res.arrays
        assert PRESYN in batch_res.graphs

        # Check that request of only one source also works.
        request = BatchRequest()
        request.add(PRESYN, window_request)
        batch_res = pipeline.request_batch(request)
        assert GT_LABELS not in batch_res.arrays
        assert PRESYN in batch_res.graphs

    # Check that it fails, when having two sources that provide the same type.
    arraysource2 = ArrayTestSoure(voxel_size)
    pipeline_fail = (arraysource, arraysource2) + MergeProvider() + RandomLocation()
    with pytest.raises(PipelineSetupError):
        with build(pipeline_fail):
            pass
