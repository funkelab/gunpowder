import numpy as np
import pytest
from scipy.ndimage import center_of_mass

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    Batch,
    BatchProvider,
    BatchRequest,
    Coordinate,
    DeformAugment,
    GPGraphSource,
    GraphKey,
    GraphSpec,
    Pipeline,
    Roi,
    build,
)
from gunpowder.graph import Graph, Node


class GraphTestSource3D(BatchProvider):
    def __init__(self, graph_key: GraphKey, array_key: ArrayKey, array_key2: ArrayKey):
        self.graph_key = graph_key
        self.array_key = array_key
        self.array_key2 = array_key2

    def setup(self):
        self.nodes = [
            Node(id=1, location=np.array([0, 0.5, 0])),
            Node(id=2, location=np.array([0, 10.5, 0])),
            Node(id=3, location=np.array([0, 20.5, 0])),
            Node(id=4, location=np.array([0, 30.5, 0])),
            Node(id=5, location=np.array([0, 40.5, 0])),
            Node(id=6, location=np.array([0, 50.5, 0])),
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


@pytest.mark.parametrize("rotate", [True, False])
@pytest.mark.parametrize("spatial_dims", [2, 3])
@pytest.mark.parametrize("fast_points", [True, False])
@pytest.mark.parametrize("subsampling", [1, 2, 4])
def test_3d_basics(rotate, spatial_dims, fast_points, subsampling):
    test_labels = ArrayKey("TEST_LABELS")
    test_labels2 = ArrayKey("TEST_LABELS2")
    test_graph = GraphKey("TEST_GRAPH")

    pipeline = GraphTestSource3D(test_graph, test_labels, test_labels2) + DeformAugment(
        [4] * spatial_dims,
        [1] * spatial_dims,
        graph_raster_voxel_size=[1] * spatial_dims,
        rotate=rotate,
        spatial_dims=spatial_dims,
        use_fast_points_transform=fast_points,
        subsample=subsampling,
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

            assert Node(id=1, location=np.array([0, 0, 0])) in list(graph.nodes), list(
                graph.nodes
            )

            # graph should have moved together with the voxels
            for node in graph.nodes:
                loc = node.location
                if labels.spec.roi.contains(loc):
                    loc = (loc - labels.spec.roi.begin) / labels.spec.voxel_size
                    loc = np.array(loc)
                    com = center_of_mass(labels.data == node.id)
                    if any(np.isnan(com)):
                        # cannot assume that the rasterized data will exist after defomation
                        continue
                    assert (
                        np.linalg.norm(com - loc)
                        < np.linalg.norm(labels.spec.voxel_size) * 2
                    ), (com, loc)

                loc2 = node.location
                if labels2.spec.roi.contains(loc2):
                    loc2 = (loc2 - labels2.spec.roi.begin) / labels2.spec.voxel_size
                    loc2 = np.array(loc2)
                    com2 = center_of_mass(labels2.data == node.id)
                    assert (
                        np.linalg.norm(com2 - loc2)
                        < np.linalg.norm(labels2.spec.voxel_size) * 2
                    ), (com2, loc2)


@pytest.fixture
def mock_4d_source() -> Pipeline:
    points = GraphKey("points")
    nodes = [
        Node(0, np.array([0, 0, 0, 0])),
        Node(1, np.array([5, 10, 10, 10])),
        Node(2, np.array([10, 50, 50, 50])),
        Node(3, np.array([15, 90, 90, 90])),
    ]
    points_source = GPGraphSource(
        points, Graph(nodes=nodes, edges=[], spec=GraphSpec())
    )
    return points_source


def test_4d_basics(mock_4d_source):
    points = GraphKey("points")
    deform = DeformAugment(
        control_point_spacing=Coordinate((5, 5, 5)),
        jitter_sigma=[0.1, 0.1, 0.1],
        scale_interval=[0.9, 1.1],
        rotate=True,
        subsample=4,
        spatial_dims=3,
        use_fast_points_transform=True,
    )
    pipeline = mock_4d_source + deform

    request = BatchRequest()
    request_shape = Coordinate((15, 40, 40, 40))
    request_roi = Roi(offset=(5, 30, 30, 30), shape=request_shape)
    points_request = GraphSpec(
        request_roi,
    )
    request[points] = points_request

    with build(pipeline):
        batch = pipeline.request_batch(request)
        points_data = batch[points]
        # not enough deformation to remove node from center
        assert len(list(points_data.nodes)) == 1
