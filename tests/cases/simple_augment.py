from gunpowder import (
    Batch,
    BatchProvider,
    BatchRequest,
    Array,
    ArrayKey,
    ArraySpec,
    Graph,
    GraphKey,
    GraphSpec,
    Node,
    Coordinate,
    Roi,
    SimpleAugment,
    MergeProvider,
    build,
)

import numpy as np

from .helper_sources import GraphSource, ArraySource


def test_mirror():
    voxel_size = Coordinate((20, 20))
    graph_key = GraphKey("GRAPH")
    array_key = ArrayKey("ARRAY")
    graph = Graph(
        [Node(id=1, location=np.array([450, 550]))],
        [],
        GraphSpec(roi=Roi((100, 200), (800, 600))),
    )
    data = np.zeros([40, 30])
    data[17, 17] = 1
    array = Array(
        data, ArraySpec(roi=Roi((100, 200), (800, 600)), voxel_size=voxel_size)
    )

    default_pipeline = (
        (GraphSource(graph_key, graph), ArraySource(array_key, array))
        + MergeProvider()
        + SimpleAugment(mirror_only=[0, 1], transpose_only=[], mirror_probs=[0, 0])
    )

    mirror_pipeline = (
        (GraphSource(graph_key, graph), ArraySource(array_key, array))
        + MergeProvider()
        + SimpleAugment(mirror_only=[0, 1], transpose_only=[], mirror_probs=[1, 1])
    )

    request = BatchRequest()
    request[graph_key] = GraphSpec(roi=Roi((400, 500), (200, 300)))
    request[array_key] = ArraySpec(roi=Roi((400, 500), (200, 300)))
    with build(default_pipeline):
        expected_location = [450, 550]
        batch = default_pipeline.request_batch(request)

        assert len(list(batch[graph_key].nodes)) == 1
        node = list(batch[graph_key].nodes)[0]
        assert all(np.isclose(node.location, expected_location))
        node_voxel_index = Coordinate(
            (node.location - batch[array_key].spec.roi.offset) / voxel_size
        )
        assert batch[array_key].data[node_voxel_index] == 1

    with build(mirror_pipeline):
        expected_location = [550, 750]
        batch = mirror_pipeline.request_batch(request)

        assert len(list(batch[graph_key].nodes)) == 1
        node = list(batch[graph_key].nodes)[0]
        assert all(np.isclose(node.location, expected_location))
        node_voxel_index = Coordinate(
            (node.location - batch[array_key].spec.roi.offset) / voxel_size
        )
        assert (
            batch[array_key].data[node_voxel_index] == 1
        ), f"Node at {np.where(batch[array_key].data == 1)} not {node_voxel_index}"


def test_transpose():
    voxel_size = Coordinate((20, 20))
    graph_key = GraphKey("GRAPH")
    array_key = ArrayKey("ARRAY")
    graph = Graph(
        [Node(id=1, location=np.array([450, 550]))],
        [],
        GraphSpec(roi=Roi((100, 200), (800, 600))),
    )
    data = np.zeros([40, 30])
    data[17, 17] = 1
    array = Array(
        data, ArraySpec(roi=Roi((100, 200), (800, 600)), voxel_size=voxel_size)
    )

    default_pipeline = (
        (GraphSource(graph_key, graph), ArraySource(array_key, array))
        + MergeProvider()
        + SimpleAugment(mirror_only=[], transpose_only=[0, 1], transpose_probs=[0, 0])
    )

    transpose_pipeline = (
        (GraphSource(graph_key, graph), ArraySource(array_key, array))
        + MergeProvider()
        + SimpleAugment(mirror_only=[], transpose_only=[0, 1], transpose_probs=[1, 1])
    )

    request = BatchRequest()
    request[graph_key] = GraphSpec(roi=Roi((400, 500), (200, 300)))
    request[array_key] = ArraySpec(roi=Roi((400, 500), (200, 300)))
    with build(default_pipeline):
        expected_location = [450, 550]
        batch = default_pipeline.request_batch(request)

        assert len(list(batch[graph_key].nodes)) == 1
        node = list(batch[graph_key].nodes)[0]
        assert all(np.isclose(node.location, expected_location))
        node_voxel_index = Coordinate(
            (node.location - batch[array_key].spec.roi.offset) / voxel_size
        )
        assert (
            batch[array_key].data[node_voxel_index] == 1
        ), f"Node at {np.where(batch[array_key].data == 1)} not {node_voxel_index}"

    with build(transpose_pipeline):
        expected_location = [410, 590]
        batch = transpose_pipeline.request_batch(request)

        assert len(list(batch[graph_key].nodes)) == 1
        node = list(batch[graph_key].nodes)[0]
        assert all(np.isclose(node.location, expected_location))
        node_voxel_index = Coordinate(
            (node.location - batch[array_key].spec.roi.offset) / voxel_size
        )
        assert (
            batch[array_key].data[node_voxel_index] == 1
        ), f"Node at {np.where(batch[array_key].data == 1)} not {node_voxel_index}"


def test_mirror_and_transpose():
    voxel_size = Coordinate((20, 20))
    graph_key = GraphKey("GRAPH")
    array_key = ArrayKey("ARRAY")
    graph = Graph(
        [Node(id=1, location=np.array([450, 550]))],
        [],
        GraphSpec(roi=Roi((100, 200), (800, 600))),
    )
    data = np.zeros([40, 30])
    data[17, 17] = 1
    array = Array(
        data, ArraySpec(roi=Roi((100, 200), (800, 600)), voxel_size=voxel_size)
    )

    default_pipeline = (
        (GraphSource(graph_key, graph), ArraySource(array_key, array))
        + MergeProvider()
        + SimpleAugment(
            mirror_only=[0, 1],
            transpose_only=[0, 1],
            mirror_probs=[0, 0],
            transpose_probs={(0, 1): 1},
        )
    )

    augmented_pipeline = (
        (GraphSource(graph_key, graph), ArraySource(array_key, array))
        + MergeProvider()
        + SimpleAugment(
            mirror_only=[0, 1],
            transpose_only=[0, 1],
            mirror_probs=[0, 1],
            transpose_probs={(1, 0): 1},
        )
    )

    request = BatchRequest()
    request[graph_key] = GraphSpec(roi=Roi((400, 500), (200, 300)))
    request[array_key] = ArraySpec(roi=Roi((400, 500), (200, 300)))
    with build(default_pipeline):
        expected_location = [450, 550]
        batch = default_pipeline.request_batch(request)

        assert len(list(batch[graph_key].nodes)) == 1
        node = list(batch[graph_key].nodes)[0]
        assert all(np.isclose(node.location, expected_location))
        node_voxel_index = Coordinate(
            (node.location - batch[array_key].spec.roi.offset) / voxel_size
        )
        assert batch[array_key].data[node_voxel_index] == 1

    with build(augmented_pipeline):
        expected_location = [590, 590]
        batch = augmented_pipeline.request_batch(request)

        assert len(list(batch[graph_key].nodes)) == 1
        node = list(batch[graph_key].nodes)[0]
        assert all(np.isclose(node.location, expected_location))
        node_voxel_index = Coordinate(
            (np.array(expected_location) - batch[array_key].spec.roi.offset)
            / voxel_size
        )
        assert (
            batch[array_key].data[node_voxel_index] == 1
        ), f"Node at {np.where(batch[array_key].data == 1)} not {node_voxel_index}"


def test_mismatched_voxel_multiples():
    """
    Ensure we don't shift by half a voxel when transposing 2 axes.

    If voxel_size = [2, 2], and we transpose array of shape [4, 6]:

        center = total_roi.center -> [2, 3]

        # Get distance from center, then transpose
        dist_to_center = center - roi.offset -> [2, 3]
        dist_to_center = transpose(dist_to_center)  -> [3, 2]

        # Using the transposed distance to center, get the offset.
        new_offset = center - dist_to_center -> [-1, 1]

        shape = transpose(shape) -> [6, 4]

        original = ((0, 0), (4, 6))
        transposed = ((-1, 1), (6, 4))

    This result is what we would expect from tranposing, but no longer fits the voxel grid.
    dist_to_center should be limited to multiples of the lcm_voxel_size.

        instead we should get:
        original = ((0, 0), (4, 6))
        transposed = ((0, 0), (6, 4))
    """

    test_array = ArrayKey("TEST_ARRAY")
    data = np.zeros([3, 3])
    data[
        2, 1
    ] = 1  # voxel has Roi((4, 2) (2, 2)). Contained in Roi((0, 0), (6, 4)). at 2, 1
    source = ArraySource(
        test_array,
        Array(
            data,
            ArraySpec(roi=Roi((0, 0), (6, 6)), voxel_size=(2, 2)),
        ),
    )
    pipeline = source + SimpleAugment(
        mirror_only=[], transpose_only=[0, 1], transpose_probs={(1, 0): 1}
    )

    with build(pipeline):
        request = BatchRequest()
        request[test_array] = ArraySpec(roi=Roi((0, 0), (4, 6)))

        batch = pipeline.request_batch(request)
        data = batch[test_array].data

        assert data[1, 2] == 1, f"{data}"
