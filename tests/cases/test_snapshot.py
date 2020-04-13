import numpy as np

from gunpowder import (
    Snapshot,
    Batch,
    BatchFilter,
    BatchProvider,
    Array,
    ArrayKey,
    ArraySpec,
    Roi,
    Coordinate,
    BatchRequest,
    build,
)


class TestSource(BatchProvider):
    def setup(self):
        self.a = ArrayKey("A")
        self.b = ArrayKey("B")
        self.array_spec = ArraySpec(
            roi=Roi((0, 0, 0), (10, 10, 10)), voxel_size=Coordinate((1, 1, 1))
        )
        self.provides(self.a, self.array_spec)
        self.provides(self.b, self.array_spec)

    def provide(self, request):
        outputs = Batch()
        for array_key, spec in request.array_specs.items():
            outputs[array_key] = Array(np.zeros((10, 10, 10)), self.array_spec)
        return outputs


class CheckRequest(BatchFilter):
    def setup(self):
        self.a = ArrayKey("A")
        self.b = ArrayKey("B")
        self.array_spec = ArraySpec(
            roi=Roi((0, 0, 0), (10, 10, 10)), voxel_size=Coordinate((1, 1, 1))
        )
        self.updates(self.a, self.array_spec)
        self.updates(self.b, self.array_spec)

        self.enable_autoskip()

    def prepare(self, request):
        return request

    def process(self, batch, request):
        outputs = Batch()
        if self.b in batch:
            outputs[self.b] = batch[self.b]
            outputs[self.a] = Array(batch[self.a].data + 1, batch[self.a].spec.copy())
        else:
            print(f"batch a: {batch[self.a]}")
            outputs[self.a] = batch[self.a]
        return outputs


def test_snapshot():
    a = ArrayKey("A")
    b = ArrayKey("B")
    spec = ArraySpec(roi=Roi((0, 0, 0), (10, 10, 10)))

    request = BatchRequest()
    request[a] = spec

    snapshot_request = BatchRequest()
    snapshot_request[a] = spec
    snapshot_request[b] = spec

    pipeline = (
        TestSource()
        + CheckRequest()
        + Snapshot(
            {},
            every=2,
            additional_request=snapshot_request,
        )
    )

    with build(pipeline):
        batch_0 = pipeline.request_batch(request)
        assert batch_0[a].data.sum() == 1000
        batch_1 = pipeline.request_batch(request)
        assert batch_1[a].data.sum() == 0
        batch_2 = pipeline.request_batch(request)
        assert batch_2[a].data.sum() == 1000
        batch_3 = pipeline.request_batch(request)
        assert batch_3[a].data.sum() == 0
