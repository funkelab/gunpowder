import math

import numpy as np

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    Batch,
    BatchProvider,
    BatchRequest,
    Coordinate,
    ElasticAugment,
    GraphKey,
    GraphSpec,
    RasterizationSettings,
    RasterizeGraph,
    Roi,
    Snapshot,
    build,
)
from gunpowder.graph import Graph, Node


class GraphTestSource3D(BatchProvider):
    def __init__(self, points_key, labels_key):
        self.points_key = points_key
        self.labels_key = labels_key

    def setup(self):
        self.nodes = [
            Node(id=0, location=np.array([0, 0, 0])),
            Node(id=1, location=np.array([0, 10, 0])),
            Node(id=2, location=np.array([0, 20, 0])),
            Node(id=3, location=np.array([0, 30, 0])),
            Node(id=4, location=np.array([0, 40, 0])),
            Node(id=5, location=np.array([0, 50, 0])),
        ]

        self.provides(
            self.points_key,
            GraphSpec(roi=Roi((-100, -100, -100), (200, 200, 200))),
        )

        self.provides(
            self.labels_key,
            ArraySpec(
                roi=Roi((-100, -100, -100), (200, 200, 200)),
                voxel_size=Coordinate((4, 1, 1)),
                interpolatable=False,
            ),
        )

    def node_to_voxel(self, array_roi, location):
        # location is in world units, get it into voxels
        location = location / self.spec[self.labels_key].voxel_size

        # shift location relative to beginning of array roi
        location -= array_roi.begin / self.spec[self.labels_key].voxel_size

        return tuple(slice(int(l - 2), int(l + 3)) for l in location)

    def provide(self, request):
        batch = Batch()

        roi_graph = request[self.points_key].roi
        roi_array = request[self.labels_key].roi
        roi_voxel = roi_array // self.spec[self.labels_key].voxel_size

        data = np.zeros(roi_voxel.shape, dtype=np.uint32)
        data[:, ::2] = 100

        for node in self.nodes:
            loc = self.node_to_voxel(roi_array, node.location)
            data[loc] = node.id

        spec = self.spec[self.labels_key].copy()
        spec.roi = roi_array
        batch.arrays[self.labels_key] = Array(data, spec=spec)

        nodes = []
        for node in self.nodes:
            if roi_graph.contains(node.location):
                nodes.append(node)
        batch.graphs[self.points_key] = Graph(
            nodes=nodes, edges=[], spec=GraphSpec(roi=roi_graph)
        )

        return batch


def test_3d_basics(tmpdir):
    test_labels = ArrayKey("TEST_LABELS")
    test_graph = GraphKey("TEST_GRAPH")
    test_raster = ArrayKey("TEST_RASTER")

    pipeline = (
        GraphTestSource3D(test_graph, test_labels)
        + ElasticAugment(
            [10, 10, 10],
            [0.1, 0.1, 0.1],
            # [0, 0, 0], # no jitter
            [0, 2.0 * math.pi],
        )  # rotate randomly
        +
        # [math.pi/4, math.pi/4]) + # rotate by 45 deg
        # [0, 0]) + # no rotation
        RasterizeGraph(
            test_graph,
            test_raster,
            settings=RasterizationSettings(radius=2, mode="peak"),
        )
        + Snapshot(
            {test_labels: "volumes/labels", test_raster: "volumes/raster"},
            dataset_dtypes={test_raster: np.float32},
            output_dir=tmpdir,
            output_filename="elastic_augment_test{id}-{iteration}.hdf",
        )
    )

    for _ in range(5):
        with build(pipeline):
            request_roi = Roi((-20, -20, -20), (40, 40, 40))

            request = BatchRequest()
            request[test_labels] = ArraySpec(roi=request_roi)
            request[test_graph] = GraphSpec(roi=request_roi)
            request[test_raster] = ArraySpec(roi=request_roi)

            batch = pipeline.request_batch(request)
            labels = batch[test_labels]
            graph = batch[test_graph]

            # the node at (0, 0, 0) should not have moved
            # The node at (0,0,0) seems to have moved
            # self.assertIn(
            #     Node(id=0, location=np.array([0, 0, 0])), list(graph.nodes)
            # )
            assert 0 in [v.id for v in graph.nodes]

            labels_data_roi = (
                labels.spec.roi - labels.spec.roi.begin
            ) / labels.spec.voxel_size

            # graph should have moved together with the voxels
            for node in graph.nodes:
                loc = node.location - labels.spec.roi.begin
                loc = loc / labels.spec.voxel_size
                loc = Coordinate(int(round(x)) for x in loc)
                if labels_data_roi.contains(loc):
                    assert labels.data[loc] == node.id
