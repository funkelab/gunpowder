from gunpowder import (
    BatchProvider,
    GraphSpec,
    Roi,
    Coordinate,
    ArrayKeys,
    ArraySpec,
    Batch,
    Array,
    ArrayKey,
    GraphKey,
    BatchRequest,
    RasterizationSettings,
    RasterizeGraph,
    DeformAugment,
    build,
)
from gunpowder.graph import GraphKeys, Graph, Node
from .provider_test import ProviderTest

import numpy as np
import math


class GraphTestSource3D(BatchProvider):
    def __init__(self, graph_key: GraphKey, array_key: ArrayKey, array_key2: ArrayKey):
        self.graph_key = graph_key
        self.array_key = array_key
        self.array_key2 = array_key2

    def setup(self):
        self.nodes = [
            Node(id=1, location=np.array([0, 0, 0])),
            Node(id=2, location=np.array([0, 10, 0])),
            Node(id=3, location=np.array([0, 20, 0])),
            Node(id=4, location=np.array([0, 30, 0])),
            Node(id=5, location=np.array([0, 40, 0])),
            Node(id=6, location=np.array([0, 50, 0])),
        ]

        self.provides(
            self.graph_key,
            GraphSpec(roi=Roi((-100, -100, -100), (200, 200, 200))),
        )

        self.provides(
            self.array_key,
            ArraySpec(
                roi=Roi((-100, -100, -100), (200, 200, 200)),
                voxel_size=Coordinate((4, 1, 1)),
                interpolatable=False,
            ),
        )

        self.provides(
            self.array_key2,
            ArraySpec(
                roi=Roi((-100, -100, -100), (200, 200, 200)),
                voxel_size=Coordinate((1, 2, 1)),
                interpolatable=False,
            ),
        )

    def node_to_voxel(self, array_roi, voxel_size, location):
        # location is in world units, get it into voxels
        location = location / voxel_size

        # shift location relative to beginning of array roi
        location -= array_roi.begin / voxel_size

        return tuple(slice(int(l - 1), int(l + 2)) for l in location)

    def provide(self, request):
        batch = Batch()

        roi_graph = request[self.graph_key].roi
        roi_voxel = request[self.array_key].roi // self.spec[self.array_key].voxel_size

        data = np.zeros(
            (request[self.array_key].roi // self.spec[self.array_key].voxel_size).shape,
            dtype=np.uint32,
        )

        for node in self.nodes:
            loc = self.node_to_voxel(
                request[self.array_key].roi,
                self.spec[self.array_key].voxel_size,
                node.location,
            )
            data[loc] = node.id

        data2 = np.zeros(
            (
                request[self.array_key2].roi // self.spec[self.array_key2].voxel_size
            ).shape,
            dtype=np.uint32,
        )

        for node in self.nodes:
            loc = self.node_to_voxel(
                request[self.array_key2].roi,
                self.spec[self.array_key2].voxel_size,
                node.location,
            )
            data2[loc] = node.id

        spec = self.spec[self.array_key].copy()
        spec.roi = request[self.array_key].roi
        batch.arrays[self.array_key] = Array(data, spec=spec)

        spec2 = self.spec[self.array_key2].copy()
        spec2.roi = request[self.array_key2].roi
        batch.arrays[self.array_key2] = Array(data2, spec=spec2)

        nodes = []
        for node in self.nodes:
            if roi_graph.contains(node.location):
                nodes.append(node)
        batch.graphs[self.graph_key] = Graph(
            nodes=nodes, edges=[], spec=GraphSpec(roi=roi_graph)
        )

        return batch


def test_3d_basics():
    test_labels = ArrayKey("TEST_LABELS")
    test_labels2 = ArrayKey("TEST_LABELS2")
    test_graph = GraphKey("TEST_GRAPH")

    pipeline = GraphTestSource3D(test_graph, test_labels, test_labels2) + DeformAugment(
        [10, 10, 10], [2, 2, 2], graph_raster_voxel_size=(2, 2, 2)
    )

    for _ in range(5):
        with build(pipeline):
            request_roi = Roi((-20, -20, -20), (40, 40, 40))

            request = BatchRequest()
            request[test_labels] = ArraySpec(roi=request_roi)
            request[test_labels2] = ArraySpec(roi=request_roi / 2)
            request[test_graph] = GraphSpec(roi=request_roi)

            batch = pipeline.request_batch(request)
            labels = batch[test_labels]
            labels2 = batch[test_labels2]
            graph = batch[test_graph]

            assert Node(id=1, location=np.array([0, 0, 0])) in list(graph.nodes)

            # graph should have moved together with the voxels
            for node in graph.nodes:

                loc = node.location - labels.spec.roi.begin
                if labels.spec.roi.contains(loc):
                    loc = loc / labels.spec.voxel_size
                    loc = Coordinate(int(round(x)) for x in loc)
                    assert labels.data[loc] == node.id

                loc2 = node.location - labels2.spec.roi.begin
                if labels2.spec.roi.contains(loc2):
                    loc2 = loc2 / labels2.spec.voxel_size
                    loc2 = Coordinate(int(round(x)) for x in loc2)
                    assert labels2.data[loc2] == node.id
