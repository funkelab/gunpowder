import numpy as np

from gunpowder import (
    BatchProvider,
    BatchFilter,
    Array,
    ArraySpec,
    ArrayKey,
    Graph,
    GraphSpec,
    GraphKey,
    Batch,
    BatchRequest,
    Roi,
    PipelineRequestError,
    build,
)
import pytest


class ArrayTestSource(BatchProvider):
    def __init__(self, key, spec):
        default_spec = ArraySpec(
            voxel_size=(1,) * spec.roi.dims,
            interpolatable=False,
            nonspatial=False,
            dtype=np.uint8,
        )
        default_spec.update_with(spec)
        spec = default_spec
        self.key = key
        self.array = Array(
            np.zeros(
                spec.roi.shape / spec.voxel_size,
                dtype=spec.dtype,
            ),
            spec=spec,
        )

    def setup(self):
        self.provides(self.key, self.array.spec)

    def provide(self, request):
        batch = Batch()
        roi = request[self.key].roi
        batch[self.key] = self.array.crop(roi)
        return batch


class RequiresSpec(BatchFilter):
    def __init__(self, key, required_spec):
        self.key = key
        self.required_spec = required_spec

    def setup(self):
        self.updates(self.key, self.spec[self.key].copy())

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.key] = self.required_spec
        return deps

    def process(self, batch, request):
        return batch


@pytest.mark.parametrize("request_dtype", [np.uint8, np.int64, np.float32])
def test_require_dtype(request_dtype):
    dtypes = [
        np.uint8,
        np.uint16,
        np.uint32,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ]
    array = ArrayKey("ARRAY")
    roi = Roi((0, 0), (10, 10))
    for dtype in dtypes:
        source = ArrayTestSource(array, ArraySpec(roi=roi, dtype=dtype))
        pipeline = source + RequiresSpec(array, ArraySpec(roi=roi, dtype=request_dtype))
        with build(pipeline):
            batch_request = BatchRequest()
            batch_request[array] = ArraySpec(roi)

            if dtype == request_dtype:
                pipeline.request_batch(batch_request)
            else:
                with pytest.raises(PipelineRequestError):
                    pipeline.request_batch(batch_request)


@pytest.mark.parametrize("request_voxel_size", [(1, 1), (2, 2)])
def test_require_voxel_size(request_voxel_size):
    voxel_sizes = [
        (1, 1),
        (4, 4),
        (6, 6),
    ]
    array = ArrayKey("ARRAY")
    roi = Roi((0, 0), (12, 12))
    for voxel_size in voxel_sizes:
        source = ArrayTestSource(array, ArraySpec(roi=roi, voxel_size=voxel_size))
        pipeline = source + RequiresSpec(
            array, ArraySpec(roi=roi, voxel_size=request_voxel_size)
        )
        with build(pipeline):
            batch_request = BatchRequest()
            batch_request[array] = ArraySpec(roi)

            if voxel_size == request_voxel_size:
                pipeline.request_batch(batch_request)
            else:
                with pytest.raises(PipelineRequestError):
                    pipeline.request_batch(batch_request)


class GraphTestSource(BatchProvider):
    def __init__(self, key, spec):
        default_spec = GraphSpec(directed=True)
        default_spec.update_with(spec)
        spec = default_spec
        self.key = key
        self.graph = Graph(
            [],
            [],
            spec=spec,
        )

    def setup(self):
        self.provides(self.key, self.graph.spec)

    def provide(self, request):
        batch = Batch()
        roi = request[self.key].roi
        batch[self.key] = self.graph.crop(roi)
        return batch


@pytest.mark.parametrize("requested_directed", [True, False])
def test_require_directed(requested_directed):
    directed_options = [True, False]
    graph = GraphKey("GRAPH")
    roi = Roi((0, 0), (10, 10))
    for provided_directed in directed_options:
        source = GraphTestSource(graph, GraphSpec(roi=roi, directed=provided_directed))
        pipeline = source + RequiresSpec(
            graph, GraphSpec(roi=roi, directed=requested_directed)
        )
        with build(pipeline):
            batch_request = BatchRequest()
            batch_request[graph] = GraphSpec(roi)

            if provided_directed == requested_directed:
                pipeline.request_batch(batch_request)
            else:
                with pytest.raises(PipelineRequestError):
                    pipeline.request_batch(batch_request)
