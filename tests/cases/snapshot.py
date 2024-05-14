from pathlib import Path

import h5py
import numpy as np

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    Batch,
    BatchProvider,
    BatchRequest,
    Coordinate,
    Graph,
    GraphKey,
    GraphKeys,
    GraphSpec,
    Roi,
    Snapshot,
    build,
)


class ExampleSource(BatchProvider):
    def __init__(self, keys, specs, every=2):
        self.keys = keys
        self.specs = specs
        self.n = 0
        self.every = every

    def setup(self):
        for key, spec in zip(self.keys, self.specs):
            self.provides(key, spec)

    def provide(self, request):
        outputs = Batch()
        if self.n % self.every == 0:
            assert GraphKeys.TEST_GRAPH in request
        else:
            assert GraphKeys.TEST_GRAPH not in request
        for key, spec in request.items():
            if isinstance(key, GraphKey):
                outputs[key] = Graph([], [], spec)
            if isinstance(key, ArrayKey):
                spec.voxel_size = self.spec[key].voxel_size
                outputs[key] = Array(np.zeros(spec.roi.shape, dtype=spec.dtype), spec)
        self.n += 1
        return outputs


def test_3d(tmpdir):
    test_graph = GraphKey("TEST_GRAPH")
    graph_spec = GraphSpec(roi=Roi((0, 0, 0), (5, 5, 5)))
    test_array = ArrayKey("TEST_ARRAY")
    array_spec = ArraySpec(
        roi=Roi((0, 0, 0), (5, 5, 5)), voxel_size=Coordinate((1, 1, 1))
    )
    test_array2 = ArrayKey("TEST_ARRAY2")
    array2_spec = ArraySpec(
        roi=Roi((0, 0, 0), (5, 5, 5)), voxel_size=Coordinate((1, 1, 1))
    )

    snapshot_request = BatchRequest()
    snapshot_request.add(test_graph, Coordinate((5, 5, 5)))

    pipeline = ExampleSource(
        [test_graph, test_array, test_array2], [graph_spec, array_spec, array2_spec]
    ) + Snapshot(
        {
            test_graph: "graphs/graph",
            test_array: "volumes/array",
            test_array2: "volumes/array2",
        },
        output_dir=tmpdir,
        every=2,
        additional_request=snapshot_request,
        output_filename="snapshot.hdf",
    )

    snapshot_file_path = Path(tmpdir, "snapshot.hdf")

    with build(pipeline):
        request = BatchRequest()
        roi = Roi((0, 0, 0), (5, 5, 5))

        request[test_array] = ArraySpec(roi=roi)
        request[test_array2] = ArraySpec(roi=roi)

        pipeline.request_batch(request)

        assert snapshot_file_path.exists()
        f = h5py.File(snapshot_file_path, "r+")
        assert f["volumes/array"] is not None
        assert f["graphs/graph-ids"] is not None

        snapshot_file_path.unlink()

        pipeline.request_batch(request)

        assert not snapshot_file_path.exists()
