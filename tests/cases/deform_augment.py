import numpy as np
import dask.array as da
import pytest
from scipy.ndimage import center_of_mass
from funlib.persistence import Array

from gunpowder import (
    ArrayKey,
    ArraySpec,
    Batch,
    BatchProvider,
    BatchRequest,
    Coordinate,
    DeformAugment,
    GPGraphSource,
    ArraySource,
    GraphKey,
    GraphSpec,
    Pipeline,
    Roi,
    build,
    MergeProvider,
)
from gunpowder.graph import Graph, Node


@pytest.fixture
def mock_3d_source():
    def node_to_voxel(array_roi, voxel_size, location):
        location = location / voxel_size
        location -= array_roi.begin / voxel_size
        return tuple(slice(int(l - 1), int(l + 2)) for l in location)

    nodes = [
        Node(id=1, location=np.array([0, 0.5, 0])),
        Node(id=2, location=np.array([0, 10.5, 0])),
        Node(id=3, location=np.array([0, 20.5, 0])),
        Node(id=4, location=np.array([0, 30.5, 0])),
        Node(id=5, location=np.array([0, 40.5, 0])),
        Node(id=6, location=np.array([0, 50.5, 0])),
    ]
    g = Graph(
        nodes, edges=[], spec=GraphSpec(roi=Roi((-100, -100, -100), (200, 200, 200)))
    )

    array_spec1 = ArraySpec(
        roi=Roi((-100, -100, -100), (200, 200, 200)),
        voxel_size=Coordinate((4, 1, 1)),
        interpolatable=False,
    )
    data1 = da.zeros((array_spec1.roi.shape // array_spec1.voxel_size), dtype=np.uint32)
    array_spec2 = ArraySpec(
        roi=Roi((-100, -100, -100), (200, 200, 200)),
        voxel_size=Coordinate((1, 2, 1)),
        interpolatable=False,
    )
    data2 = da.zeros((array_spec2.roi.shape // array_spec2.voxel_size), dtype=np.uint32)
    for node in nodes:
        loc = node_to_voxel(array_spec1.roi, array_spec1.voxel_size, node.location)
        data1[loc] = node.id
        loc = node_to_voxel(array_spec2.roi, array_spec2.voxel_size, node.location)
        data2[loc] = node.id

    array_1 = Array(
        data1, offset=array_spec1.roi.offset, voxel_size=array_spec1.voxel_size
    )
    array_2 = Array(
        data2, offset=array_spec2.roi.offset, voxel_size=array_spec2.voxel_size
    )

    return (
        GPGraphSource(GraphKey("G"), g),
        ArraySource(ArrayKey("A1"), array_1),
        ArraySource(ArrayKey("A2"), array_2),
    ) + MergeProvider()


@pytest.mark.parametrize("rotate", [True, False])
@pytest.mark.parametrize("spatial_dims", [2, 3])
@pytest.mark.parametrize("rotation_axes", [None, (1, 0), (0, 2)])
@pytest.mark.parametrize("fast_points", [True, False])
@pytest.mark.parametrize("subsampling", [1, 2, 4])
def test_3d_basics(
    mock_3d_source, rotate, spatial_dims, rotation_axes, fast_points, subsampling
):
    test_labels = ArrayKey("A1")
    test_labels2 = ArrayKey("A2")
    test_graph = GraphKey("G")

    if spatial_dims <= max(rotation_axes or [0]):
        pytest.skip("Rotation axes must be less than spatial dimensions")
    pipeline = mock_3d_source + DeformAugment(
        [4] * spatial_dims,
        [1] * spatial_dims,
        graph_raster_voxel_size=[1] * spatial_dims,
        rotate=rotate,
        spatial_dims=spatial_dims,
        use_fast_points_transform=fast_points,
        subsample=subsampling,
        rotation_axes=rotation_axes,
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
