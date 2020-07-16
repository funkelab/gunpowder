from .provider_test import ProviderTest
from gunpowder import (
    BatchProvider,
    BatchRequest,
    Batch,
    ArrayKeys,
    ArraySpec,
    Array,
    GraphKeys,
    GraphSpec,
    Graph,
    Node,
    Roi,
    Scan,
    build,
)
import numpy as np
import itertools


def coordinate_to_id(i, j, k):
    i, j, k = (i-20000) // 100, (j-2000) // 10, (k-2000) // 10
    return i + j * 20 + k * 400

class ScanTestSource(BatchProvider):

    def setup(self):

        self.provides(
            ArrayKeys.RAW,
            ArraySpec(
                roi=Roi((20000, 2000, 2000), (2000, 200, 200)),
                voxel_size=(20, 2, 2)))
        self.provides(
            ArrayKeys.GT_LABELS,
            ArraySpec(
                roi=Roi((20100,2010,2010), (1800,180,180)),
                voxel_size=(20, 2, 2)))
        self.provides(
            GraphKeys.GT_GRAPH,
            GraphSpec(
                roi=Roi((None, None, None), (None, None, None)),
            )
        )

    def provide(self, request):

        # print("ScanTestSource: Got request " + str(request))

        batch = Batch()

        # have the pixels encode their position
        for (array_key, spec) in request.array_specs.items():

            roi = spec.roi
            roi_voxel = roi // self.spec[array_key].voxel_size
            # print("ScanTestSource: Adding " + str(array_key))

            # the z,y,x coordinates of the ROI
            meshgrids = np.meshgrid(
                    range(roi_voxel.get_begin()[0], roi_voxel.get_end()[0]),
                    range(roi_voxel.get_begin()[1], roi_voxel.get_end()[1]),
                    range(roi_voxel.get_begin()[2], roi_voxel.get_end()[2]), indexing='ij')
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            # print("Roi is: " + str(roi))

            spec = self.spec[array_key].copy()
            spec.roi = roi
            batch.arrays[array_key] = Array(
                    data,
                    spec)
        
        for graph_key, spec in request.graph_specs.items():
            # node at x, y, z if x%100==0, y%10==0, z%10==0
            nodes = []
            start = spec.roi.get_begin() - tuple(x % s for x, s in zip(spec.roi.get_begin(), [100,10,10]))
            for i, j, k in itertools.product(
                *[
                    range(a, b, s)
                    for a, b, s in zip(start, spec.roi.get_end(), [100, 10, 10])
                ]
            ):
                location = np.array([i, j, k])
                if spec.roi.contains(location):
                    nodes.append(Node(id=coordinate_to_id(i, j, k), location=location))
            batch.graphs[graph_key] = Graph(
                nodes, [], spec
            )


        return batch

class TestScan(ProviderTest):

    def test_output(self):

        source = ScanTestSource()

        chunk_request = BatchRequest()
        chunk_request.add(ArrayKeys.RAW, (400,30,34))
        chunk_request.add(ArrayKeys.GT_LABELS, (200,10,14))
        chunk_request.add(GraphKeys.GT_GRAPH, (400, 30, 34))

        pipeline = ScanTestSource() + Scan(chunk_request, num_workers=10)

        with build(pipeline):

            raw_spec = pipeline.spec[ArrayKeys.RAW]
            labels_spec = pipeline.spec[ArrayKeys.GT_LABELS]
            graph_spec = pipeline.spec[GraphKeys.GT_GRAPH]

            full_request = BatchRequest({
                    ArrayKeys.RAW: raw_spec,
                    ArrayKeys.GT_LABELS: labels_spec,
                    GraphKeys.GT_GRAPH: graph_spec,
                }
            )

            batch = pipeline.request_batch(full_request)
            voxel_size = pipeline.spec[ArrayKeys.RAW].voxel_size

        # assert that pixels encode their position
        for (array_key, array) in batch.arrays.items():

            # the z,y,x coordinates of the ROI
            roi = array.spec.roi
            meshgrids = np.meshgrid(
                    range(roi.get_begin()[0]//voxel_size[0], roi.get_end()[0]//voxel_size[0]),
                    range(roi.get_begin()[1]//voxel_size[1], roi.get_end()[1]//voxel_size[1]),
                    range(roi.get_begin()[2]//voxel_size[2], roi.get_end()[2]//voxel_size[2]), indexing='ij')
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            self.assertTrue((array.data == data).all())

        for (graph_key, graph) in batch.graphs.items():

            roi = graph.spec.roi
            for i, j, k in itertools.product(range(20000, 22000, 100), range(2000, 2200, 10), range(2000, 2200, 10)):
                assert all(np.isclose(graph.node(coordinate_to_id(i, j, k)).location, np.array([i, j, k])))

        assert(batch.arrays[ArrayKeys.RAW].spec.roi.get_offset() == (20000, 2000, 2000))

        # test scanning with empty request

        pipeline = ScanTestSource() + Scan(chunk_request, num_workers=1)
        with build(pipeline):
            batch = pipeline.request_batch(BatchRequest())
