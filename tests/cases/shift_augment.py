import random

import h5py
import numpy as np
import pytest

from gunpowder import (
    ArrayKey,
    ArraySpec,
    BatchRequest,
    Coordinate,
    CsvPointsSource,
    Graph,
    GraphKey,
    GraphSpec,
    Hdf5Source,
    MergeProvider,
    Node,
    RandomLocation,
    Roi,
    ShiftAugment,
    build,
)
from gunpowder.pipeline import PipelineRequestError


# automatically set the seed for all tests
@pytest.fixture(autouse=True)
def seeds():
    random.seed(12345)
    np.random.seed(12345)


@pytest.fixture
def test_points(tmpdir):
    random.seed(1234)
    np.random.seed(1234)

    fake_points_file = tmpdir / "shift_test.csv"
    fake_data_file = tmpdir / "shift_test.hdf5"
    fake_data = np.array([[i + j for i in range(100)] for j in range(100)])
    with h5py.File(fake_data_file, "w") as f:
        f.create_dataset("testdata", shape=fake_data.shape, data=fake_data)
    fake_points = np.random.randint(0, 100, size=(2, 2))
    with open(fake_points_file, "w") as f:
        for point in fake_points:
            f.write(str(point[0]) + "\t" + str(point[1]) + "\n")

    # This fixture will run after seeds since it is set
    # with autouse=True. So make sure to reset the seeds properly at the end
    # of this fixture
    random.seed(12345)
    np.random.seed(12345)

    yield fake_points_file, fake_data_file, fake_points, fake_data


def test_prepare1(test_points):
    _, fake_data_file, *_ = test_points

    key = ArrayKey("TEST_ARRAY")
    spec = ArraySpec(voxel_size=Coordinate((1, 1)), interpolatable=True)

    hdf5_source = Hdf5Source(fake_data_file, {key: "testdata"}, array_specs={key: spec})

    request = BatchRequest()
    shape = Coordinate((3, 3))
    request.add(key, shape, voxel_size=Coordinate((1, 1)))

    shift_node = ShiftAugment(sigma=1, shift_axis=0)
    with build((hdf5_source + shift_node)):
        shift_node.prepare(request)
        assert shift_node.ndim == 2
        assert shift_node.shift_sigmas == tuple([0.0, 1.0])


def test_prepare2(test_points):
    _, fake_data_file, *_ = test_points

    key = ArrayKey("TEST_ARRAY")
    spec = ArraySpec(voxel_size=Coordinate((1, 1)), interpolatable=True)

    hdf5_source = Hdf5Source(fake_data_file, {key: "testdata"}, array_specs={key: spec})

    request = BatchRequest()
    shape = Coordinate((3, 3))
    request.add(key, shape)

    shift_node = ShiftAugment(sigma=1, shift_axis=0)

    with build((hdf5_source + shift_node)):
        shift_node.prepare(request)
        assert shift_node.ndim == 2
        assert shift_node.shift_sigmas == tuple([0.0, 1.0])


def test_pipeline1(test_points):
    _, fake_data_file, *_ = test_points

    key = ArrayKey("TEST_ARRAY")
    spec = ArraySpec(voxel_size=Coordinate((2, 1)), interpolatable=True)

    hdf5_source = Hdf5Source(fake_data_file, {key: "testdata"}, array_specs={key: spec})

    request = BatchRequest()
    shape = Coordinate((3, 3))
    request.add(key, shape, voxel_size=Coordinate((3, 1)))

    shift_node = ShiftAugment(prob_slip=0.2, prob_shift=0.2, sigma=1, shift_axis=0)
    with build((hdf5_source + shift_node)) as b:
        with pytest.raises(PipelineRequestError):
            b.request_batch(request)


def test_pipeline2(test_points):
    _, fake_data_file, *_ = test_points

    key = ArrayKey("TEST_ARRAY")
    spec = ArraySpec(voxel_size=Coordinate((3, 1)), interpolatable=True)

    hdf5_source = Hdf5Source(fake_data_file, {key: "testdata"}, array_specs={key: spec})

    request = BatchRequest()
    shape = Coordinate((3, 3))
    request[key] = ArraySpec(roi=Roi((9, 9), shape), voxel_size=Coordinate((3, 1)))

    shift_node = ShiftAugment(prob_slip=0.2, prob_shift=0.2, sigma=1, shift_axis=0)
    with build((hdf5_source + shift_node)) as b:
        b.request_batch(request)


def test_pipeline3(test_points):
    fake_points_file, fake_data_file, fake_points, fake_data = test_points

    array_key = ArrayKey("TEST_ARRAY")
    points_key = GraphKey("TEST_POINTS")
    voxel_size = Coordinate((1, 1))
    spec = ArraySpec(voxel_size=voxel_size, interpolatable=True)

    hdf5_source = Hdf5Source(
        fake_data_file, {array_key: "testdata"}, array_specs={array_key: spec}
    )
    csv_source = CsvPointsSource(
        fake_points_file,
        points_key,
        GraphSpec(roi=Roi(shape=Coordinate((100, 100)), offset=(0, 0))),
    )

    request = BatchRequest()
    shape = Coordinate((60, 60))
    request.add(array_key, shape, voxel_size=Coordinate((1, 1)))
    request.add(points_key, shape)

    shift_node = ShiftAugment(prob_slip=0.2, prob_shift=0.2, sigma=4, shift_axis=0)
    pipeline = (
        (hdf5_source, csv_source)
        + MergeProvider()
        + RandomLocation(ensure_nonempty=points_key)
        + shift_node
    )
    with build(pipeline) as b:
        request = b.request_batch(request)
        # print(request[points_key])

    target_vals = [fake_data[point[0]][point[1]] for point in fake_points]
    result_data = request[array_key].data
    result_points = list(request[points_key].nodes)
    result_vals = [
        result_data[int(point.location[0])][int(point.location[1])]
        for point in result_points
    ]

    for result_val in result_vals:
        assert result_val in target_vals


##################
# shift_and_crop #
##################


def test_shift_and_crop_static():
    shift_node = ShiftAugment(sigma=1, shift_axis=0)
    shift_node.ndim = 2
    upstream_arr = np.arange(16).reshape(4, 4)
    sub_shift_array = np.zeros(8, dtype=int).reshape(4, 2)
    roi_shape = (4, 4)
    voxel_size = Coordinate((1, 1))

    downstream_arr = np.arange(16).reshape(4, 4)

    result = shift_node.shift_and_crop(
        upstream_arr, roi_shape, sub_shift_array, voxel_size
    )
    assert np.array_equal(result, downstream_arr)


def test_shift_and_crop1():
    shift_node = ShiftAugment(sigma=1, shift_axis=0)
    shift_node.ndim = 2
    upstream_arr = np.arange(16).reshape(4, 4)
    sub_shift_array = np.zeros(8, dtype=int).reshape(4, 2)
    sub_shift_array[:, 1] = np.array([0, -1, 1, 0], dtype=int)
    roi_shape = (4, 2)
    voxel_size = Coordinate((1, 1))

    downstream_arr = np.array([[1, 2], [6, 7], [8, 9], [13, 14]], dtype=int)

    result = shift_node.shift_and_crop(
        upstream_arr, roi_shape, sub_shift_array, voxel_size
    )
    assert np.array_equal(result, downstream_arr)


def test_shift_and_crop2():
    shift_node = ShiftAugment(sigma=1, shift_axis=0)
    shift_node.ndim = 2
    upstream_arr = np.arange(16).reshape(4, 4)
    sub_shift_array = np.zeros(8, dtype=int).reshape(4, 2)
    sub_shift_array[:, 1] = np.array([0, -1, -2, 0], dtype=int)
    roi_shape = (4, 2)
    voxel_size = Coordinate((1, 1))

    downstream_arr = np.array([[0, 1], [5, 6], [10, 11], [12, 13]], dtype=int)

    result = shift_node.shift_and_crop(
        upstream_arr, roi_shape, sub_shift_array, voxel_size
    )
    assert np.array_equal(result, downstream_arr)


def test_shift_and_crop3():
    shift_node = ShiftAugment(sigma=1, shift_axis=1)
    shift_node.ndim = 2
    upstream_arr = np.arange(16).reshape(4, 4)
    sub_shift_array = np.zeros(8, dtype=int).reshape(4, 2)
    sub_shift_array[:, 0] = np.array([0, 1, 0, 2], dtype=int)
    roi_shape = (2, 4)
    voxel_size = Coordinate((1, 1))

    downstream_arr = np.array([[8, 5, 10, 3], [12, 9, 14, 7]], dtype=int)

    result = shift_node.shift_and_crop(
        upstream_arr, roi_shape, sub_shift_array, voxel_size
    )
    # print(result)
    assert np.array_equal(result, downstream_arr)


def test_shift_and_crop4():
    shift_node = ShiftAugment(sigma=1, shift_axis=1)
    shift_node.ndim = 2
    upstream_arr = np.arange(16).reshape(4, 4)
    sub_shift_array = np.zeros(8, dtype=int).reshape(4, 2)
    sub_shift_array[:, 0] = np.array([0, 2, 0, 4], dtype=int)
    roi_shape = (4, 4)
    voxel_size = Coordinate((2, 1))

    downstream_arr = np.array([[8, 5, 10, 3], [12, 9, 14, 7]], dtype=int)

    result = shift_node.shift_and_crop(
        upstream_arr, roi_shape, sub_shift_array, voxel_size
    )
    # print(result)
    assert np.array_equal(result, downstream_arr)

    result = shift_node.shift_and_crop(
        upstream_arr, roi_shape, sub_shift_array, voxel_size
    )
    # print(result)
    assert np.array_equal(result, downstream_arr)


##################
# shift_points   #
##################


def points_equal(vertices1, vertices2):
    vs1 = sorted(list(vertices1), key=lambda v: tuple(v.location))
    vs2 = sorted(list(vertices2), key=lambda v: tuple(v.location))

    for v1, v2 in zip(vs1, vs2):
        if not v1.id == v2.id:
            print(f"{vs1}, {vs2}")
            return False
        if not all(np.isclose(v1.location, v2.location)):
            print(f"{vs1}, {vs2}")
            return False
    return True


def test_points_equal():
    points1 = [Node(id=1, location=np.array([0, 1]))]
    points2 = [Node(id=1, location=np.array([0, 1]))]
    assert points_equal(points1, points2)

    points1 = [Node(id=2, location=np.array([1, 2]))]
    points2 = [Node(id=2, location=np.array([2, 1]))]
    assert not points_equal(points1, points2)


def test_shift_points1():
    data = [Node(id=1, location=np.array([0, 1]))]
    spec = GraphSpec(Roi(offset=(0, 0), shape=(5, 5)))
    points = Graph(data, [], spec)
    request_roi = Roi(offset=(0, 1), shape=(5, 3))
    shift_array = np.array([[0, -1], [0, -1], [0, 0], [0, 0], [0, 1]], dtype=int)
    lcm_voxel_size = Coordinate((1, 1))

    shifted_points = Graph([], [], GraphSpec(request_roi))
    result = ShiftAugment.shift_points(
        points,
        request_roi,
        shift_array,
        shift_axis=0,
        lcm_voxel_size=lcm_voxel_size,
    )
    # print(result)
    assert points_equal(result.nodes, shifted_points.nodes)
    assert result.spec == GraphSpec(request_roi)


def test_shift_points2():
    data = [Node(id=1, location=np.array([0, 1]))]
    spec = GraphSpec(Roi(offset=(0, 0), shape=(5, 5)))
    points = Graph(data, [], spec)
    request_roi = Roi(offset=(0, 1), shape=(5, 3))
    shift_array = np.array([[0, 0], [0, -1], [0, 0], [0, 0], [0, 1]], dtype=int)
    lcm_voxel_size = Coordinate((1, 1))

    result = ShiftAugment.shift_points(
        points,
        request_roi,
        shift_array,
        shift_axis=0,
        lcm_voxel_size=lcm_voxel_size,
    )
    # print("test 2", result.data, data)
    assert points_equal(result.nodes, data)
    assert result.spec == GraphSpec(request_roi)


def test_shift_points3():
    data = [Node(id=1, location=np.array([0, 1]))]
    spec = GraphSpec(Roi(offset=(0, 0), shape=(5, 5)))
    points = Graph(data, [], spec)
    request_roi = Roi(offset=(0, 1), shape=(5, 3))
    shift_array = np.array([[0, 1], [0, -1], [0, 0], [0, 0], [0, 1]], dtype=int)
    lcm_voxel_size = Coordinate((1, 1))

    shifted_points = Graph(
        [Node(id=1, location=np.array([0, 2]))], [], GraphSpec(request_roi)
    )
    result = ShiftAugment.shift_points(
        points,
        request_roi,
        shift_array,
        shift_axis=0,
        lcm_voxel_size=lcm_voxel_size,
    )
    # print("test 3", result.data, shifted_points.data)
    assert points_equal(result.nodes, shifted_points.nodes)
    assert result.spec == GraphSpec(request_roi)


def test_shift_points4():
    data = [
        Node(id=0, location=np.array([1, 0])),
        Node(id=1, location=np.array([1, 1])),
        Node(id=2, location=np.array([1, 2])),
        Node(id=3, location=np.array([1, 3])),
        Node(id=4, location=np.array([1, 4])),
    ]
    spec = GraphSpec(Roi(offset=(0, 0), shape=(5, 5)))
    points = Graph(data, [], spec)
    request_roi = Roi(offset=(1, 0), shape=(3, 5))
    shift_array = np.array([[1, 0], [-1, 0], [0, 0], [-1, 0], [1, 0]], dtype=int)

    lcm_voxel_size = Coordinate((1, 1))
    shifted_data = [
        Node(id=0, location=np.array([2, 0])),
        Node(id=2, location=np.array([1, 2])),
        Node(id=4, location=np.array([2, 4])),
    ]
    result = ShiftAugment.shift_points(
        points,
        request_roi,
        shift_array,
        shift_axis=1,
        lcm_voxel_size=lcm_voxel_size,
    )
    # print("test 4", result.data, shifted_data)
    assert points_equal(result.nodes, shifted_data)
    assert result.spec == GraphSpec(request_roi)


def test_shift_points5():
    data = [
        Node(id=0, location=np.array([3, 0])),
        Node(id=1, location=np.array([3, 2])),
        Node(id=2, location=np.array([3, 4])),
        Node(id=3, location=np.array([3, 6])),
        Node(id=4, location=np.array([3, 8])),
    ]
    spec = GraphSpec(Roi(offset=(0, 0), shape=(15, 10)))
    points = Graph(data, [], spec)
    request_roi = Roi(offset=(3, 0), shape=(9, 10))
    shift_array = np.array([[3, 0], [-3, 0], [0, 0], [-3, 0], [3, 0]], dtype=int)

    lcm_voxel_size = Coordinate((3, 2))
    shifted_data = [
        Node(id=0, location=np.array([6, 0])),
        Node(id=2, location=np.array([3, 4])),
        Node(id=4, location=np.array([6, 8])),
    ]
    result = ShiftAugment.shift_points(
        points,
        request_roi,
        shift_array,
        shift_axis=1,
        lcm_voxel_size=lcm_voxel_size,
    )
    # print("test 4", result.data, shifted_data)
    assert points_equal(result.nodes, shifted_data)
    assert result.spec == GraphSpec(request_roi)


#######################
# get_sub_shift_array #
#######################


def test_get_sub_shift_array1():
    total_roi = Roi(offset=(0, 0), shape=(6, 6))
    item_roi = Roi(offset=(1, 2), shape=(3, 3))
    shift_array = np.arange(12).reshape(6, 2).astype(int)
    shift_axis = 1
    lcm_voxel_size = Coordinate((1, 1))

    sub_shift_array = np.array([[4, 5], [6, 7], [8, 9]], dtype=int)
    result = ShiftAugment.get_sub_shift_array(
        total_roi, item_roi, shift_array, shift_axis, lcm_voxel_size
    )
    # print(result)
    assert np.array_equal(result, sub_shift_array)


def test_get_sub_shift_array2():
    total_roi = Roi(offset=(0, 0), shape=(6, 6))
    item_roi = Roi(offset=(1, 2), shape=(3, 3))
    shift_array = np.arange(12).reshape(6, 2).astype(int)
    shift_axis = 0
    lcm_voxel_size = Coordinate((1, 1))

    sub_shift_array = np.array([[2, 3], [4, 5], [6, 7]], dtype=int)
    result = ShiftAugment.get_sub_shift_array(
        total_roi, item_roi, shift_array, shift_axis, lcm_voxel_size
    )
    assert np.array_equal(result, sub_shift_array)


def test_get_sub_shift_array3():
    total_roi = Roi(offset=(0, 0), shape=(18, 12))
    item_roi = Roi(offset=(3, 4), shape=(9, 6))
    shift_array = np.arange(12).reshape(6, 2).astype(int)
    shift_axis = 0
    lcm_voxel_size = Coordinate((3, 2))

    sub_shift_array = np.array([[2, 3], [4, 5], [6, 7]], dtype=int)
    result = ShiftAugment.get_sub_shift_array(
        total_roi, item_roi, shift_array, shift_axis, lcm_voxel_size
    )
    # print(result)
    assert np.array_equal(result, sub_shift_array)


################################
# construct_global_shift_array #
################################


def test_construct_global_shift_array_static():
    shift_axis_len = 5
    shift_sigmas = (0.0, 1.0)
    prob_slip = 0
    prob_shift = 0
    lcm_voxel_size = Coordinate((1, 1))

    shift_array = np.zeros(shape=(shift_axis_len, len(shift_sigmas)), dtype=int)
    result = ShiftAugment.construct_global_shift_array(
        shift_axis_len, shift_sigmas, prob_shift, prob_slip, lcm_voxel_size
    )
    assert np.array_equal(result, shift_array)


def test_construct_global_shift_array1():
    shift_axis_len = 5
    shift_sigmas = (0.0, 1.0)
    prob_slip = 1
    prob_shift = 0
    lcm_voxel_size = Coordinate((1, 1))

    shift_array = np.array([[0, 0], [0, -1], [0, 1], [0, 0], [0, 1]], dtype=int)
    result = ShiftAugment.construct_global_shift_array(
        shift_axis_len, shift_sigmas, prob_slip, prob_shift, lcm_voxel_size
    )
    # print(result)
    assert len(result) == shift_axis_len
    for position_shift in result:
        assert position_shift[0] == 0
    assert np.array_equal(shift_array, result)


def test_construct_global_shift_array2():
    shift_axis_len = 5
    shift_sigmas = (0.0, 1.0)
    prob_slip = 0
    prob_shift = 1
    lcm_voxel_size = Coordinate((1, 1))

    shift_array = np.array([[0, 0], [0, -1], [0, 0], [0, 0], [0, 1]], dtype=int)
    result = ShiftAugment.construct_global_shift_array(
        shift_axis_len, shift_sigmas, prob_slip, prob_shift, lcm_voxel_size
    )
    assert len(result) == shift_axis_len
    for position_shift in result:
        assert position_shift[0] == 0
    assert np.array_equal(shift_array, result)


def test_construct_global_shift_array3():
    shift_axis_len = 5
    shift_sigmas = (0.0, 4.0)
    prob_slip = 0
    prob_shift = 1
    lcm_voxel_size = Coordinate((1, 3))

    shift_array = np.array([[0, 3], [0, 0], [0, 6], [0, 6], [0, 12]], dtype=int)
    result = ShiftAugment.construct_global_shift_array(
        shift_axis_len, shift_sigmas, prob_slip, prob_shift, lcm_voxel_size
    )
    # print(result)
    assert len(result) == shift_axis_len
    for position_shift in result:
        assert position_shift[0] == 0
    assert np.array_equal(shift_array, result)


########################
# compute_upstream_roi #
########################


def test_compute_upstream_roi_static():
    request_roi = Roi(offset=(0, 0), shape=(5, 10))
    sub_shift_array = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=int)

    upstream_roi = Roi(offset=(0, 0), shape=(5, 10))
    result = ShiftAugment.compute_upstream_roi(request_roi, sub_shift_array)
    assert upstream_roi == result


def test_compute_upstream_roi1():
    request_roi = Roi(offset=(0, 0), shape=(5, 10))
    sub_shift_array = np.array([[0, 0], [0, -1], [0, 0], [0, 0], [0, 1]], dtype=int)

    upstream_roi = Roi(offset=(0, -1), shape=(5, 12))
    result = ShiftAugment.compute_upstream_roi(request_roi, sub_shift_array)
    assert upstream_roi == result


def test_compute_upstream_roi2():
    request_roi = Roi(offset=(0, 0), shape=(5, 10))
    sub_shift_array = np.array([[2, 0], [-1, 0], [5, 0], [-2, 0], [0, 0]], dtype=int)

    upstream_roi = Roi(offset=(-5, 0), shape=(12, 10))
    result = ShiftAugment.compute_upstream_roi(request_roi, sub_shift_array)
    assert upstream_roi == result
