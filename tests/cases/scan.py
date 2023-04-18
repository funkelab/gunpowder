from gunpowder import (
    BatchProvider,
    BatchRequest,
    Batch,
    ArrayKeys,
    ArrayKey,
    ArraySpec,
    Array,
    GraphKey,
    GraphKeys,
    GraphSpec,
    Graph,
    Node,
    Roi,
    Coordinate,
    Scan,
    build,
)
import numpy as np
import itertools


def coordinate_to_id(i, j, k):
    i, j, k = (i - 20000) // 100, (j - 2000) // 10, (k - 2000) // 10
    return i + j * 20 + k * 400


class ScanTestSource(BatchProvider):
    def __init__(self, raw_key, gt_labels_key, gt_graph_key):
        self.raw_key = raw_key
        self.gt_labels_key = gt_labels_key
        self.gt_graph_key = gt_graph_key

    def setup(self):
        self.provides(
            self.raw_key,
            ArraySpec(
                roi=Roi((20000, 2000, 2000), (2000, 200, 200)), voxel_size=(20, 2, 2)
            ),
        )
        self.provides(
            self.gt_labels_key,
            ArraySpec(
                roi=Roi((20100, 2010, 2010), (1800, 180, 180)), voxel_size=(20, 2, 2)
            ),
        )
        self.provides(
            self.gt_graph_key,
            GraphSpec(
                roi=Roi((None, None, None), (None, None, None)),
            ),
        )

    def provide(self, request):
        # print("ScanTestSource: Got request " + str(request))

        batch = Batch()

        # have the pixels encode their position
        for array_key, spec in request.array_specs.items():
            roi = spec.roi
            roi_voxel = roi // self.spec[array_key].voxel_size
            # print("ScanTestSource: Adding " + str(array_key))

            # the z,y,x coordinates of the ROI
            meshgrids = np.meshgrid(
                range(roi_voxel.begin[0], roi_voxel.end[0]),
                range(roi_voxel.begin[1], roi_voxel.end[1]),
                range(roi_voxel.begin[2], roi_voxel.end[2]),
                indexing="ij",
            )
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            # print("Roi is: " + str(roi))

            spec = self.spec[array_key].copy()
            spec.roi = roi
            batch.arrays[array_key] = Array(data, spec)

        for graph_key, spec in request.graph_specs.items():
            # node at x, y, z if x%100==0, y%10==0, z%10==0
            nodes = []
            start = spec.roi.begin - (spec.roi.begin % Coordinate(100, 10, 10))
            for i, j, k in itertools.product(
                *[range(a, b, s) for a, b, s in zip(start, spec.roi.end, [100, 10, 10])]
            ):
                location = np.array([i, j, k])
                if spec.roi.contains(location):
                    nodes.append(Node(id=coordinate_to_id(i, j, k), location=location))
            batch.graphs[graph_key] = Graph(nodes, [], spec)

        return batch


def test_output():
    raw_key = ArrayKey("RAW")
    gt_labels_key = ArrayKey("GT_LABELS")
    gt_graph_key = GraphKey("GT_GRAPH")

    chunk_request = BatchRequest()
    chunk_request.add(raw_key, (400, 30, 34))
    chunk_request.add(gt_labels_key, (200, 10, 14))
    chunk_request.add(gt_graph_key, (400, 30, 34))

    pipeline = ScanTestSource(raw_key, gt_labels_key, gt_graph_key) + Scan(
        chunk_request, num_workers=10
    )

    with build(pipeline):
        raw_spec = pipeline.spec[ArrayKeys.RAW]
        labels_spec = pipeline.spec[ArrayKeys.GT_LABELS]
        graph_spec = pipeline.spec[GraphKeys.GT_GRAPH]

        full_request = BatchRequest(
            {
                ArrayKeys.RAW: raw_spec,
                ArrayKeys.GT_LABELS: labels_spec,
                GraphKeys.GT_GRAPH: graph_spec,
            }
        )

        batch = pipeline.request_batch(full_request)
        voxel_size = pipeline.spec[ArrayKeys.RAW].voxel_size

    # assert that pixels encode their position
    for array_key, array in batch.arrays.items():
        # the z,y,x coordinates of the ROI
        roi = array.spec.roi
        meshgrids = np.meshgrid(
            range(roi.begin[0] // voxel_size[0], roi.end[0] // voxel_size[0]),
            range(roi.begin[1] // voxel_size[1], roi.end[1] // voxel_size[1]),
            range(roi.begin[2] // voxel_size[2], roi.end[2] // voxel_size[2]),
            indexing="ij",
        )
        data = meshgrids[0] + meshgrids[1] + meshgrids[2]

    assert (array.data == data).all()

    for graph_key, graph in batch.graphs.items():
        roi = graph.spec.roi
        for i, j, k in itertools.product(
            range(20000, 22000, 100), range(2000, 2200, 10), range(2000, 2200, 10)
        ):
            assert all(
                np.isclose(
                    graph.node(coordinate_to_id(i, j, k)).location, np.array([i, j, k])
                )
            )

    assert batch.arrays[ArrayKeys.RAW].spec.roi.offset == (20000, 2000, 2000)

    # test scanning with empty request

    pipeline = ScanTestSource(raw_key, gt_labels_key, gt_graph_key) + Scan(
        chunk_request, num_workers=1
    )
    with build(pipeline):
        batch = pipeline.request_batch(BatchRequest())
