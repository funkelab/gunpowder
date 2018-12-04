import unittest
import numpy as np
import random
import h5py
import logging
import sys
import os
import gunpowder as gp

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class TestShiftAugment2D(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fake_points_file = "shift_test.csv"
        cls.fake_data_file = "shift_test.hdf5"
        random.seed(1234)
        np.random.seed(1234)
        cls.fake_data = np.array([[i + j for i in range(100)] for j in range(100)])
        with h5py.File(cls.fake_data_file, "w") as f:
            f.create_dataset("testdata", shape=cls.fake_data.shape, data=cls.fake_data)
        cls.fake_points = np.random.randint(0, 100, size=(2, 2))
        with open(cls.fake_points_file, "w") as f:
            for point in cls.fake_points:
                f.write(str(point[0]) + "\t" + str(point[1]) + "\n")

    def setUp(self):
        random.seed(12345)
        np.random.seed(12345)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.fake_data_file)
        os.remove(cls.fake_points_file)


    ##################
    # full pipeline  #
    ##################

    def test_prepare1(self):

        key = gp.ArrayKey("TEST_ARRAY")
        spec = gp.ArraySpec(voxel_size=gp.Coordinate((1, 1)), interpolatable=True)

        hdf5_source = gp.Hdf5Source(self.fake_data_file, {key: 'testdata'}, array_specs={key: spec})

        request = gp.BatchRequest()
        shape = gp.Coordinate((3, 3))
        request.add(key, shape, voxel_size=gp.Coordinate((1, 1)))

        shift_node = gp.ShiftAugment(sigma=1, shift_axis=0)
        with gp.build((hdf5_source + shift_node)):
            shift_node.prepare(request)
            self.assertTrue(shift_node.ndim == 2)
            self.assertTrue(shift_node.shift_sigmas == tuple([0.0, 1.0]))

    def test_prepare2(self):

        key = gp.ArrayKey("TEST_ARRAY")
        spec = gp.ArraySpec(voxel_size=gp.Coordinate((1, 1)), interpolatable=True)

        hdf5_source = gp.Hdf5Source(self.fake_data_file, {key: 'testdata'}, array_specs={key: spec})

        request = gp.BatchRequest()
        shape = gp.Coordinate((3, 3))
        request.add(key, shape)

        shift_node = gp.ShiftAugment(sigma=1, shift_axis=0)

        with gp.build((hdf5_source + shift_node)):
            shift_node.prepare(request)
            self.assertTrue(shift_node.ndim == 2)
            self.assertTrue(shift_node.shift_sigmas == tuple([0.0, 1.0]))

    def test_pipeline1(self):

        key = gp.ArrayKey("TEST_ARRAY")
        spec = gp.ArraySpec(voxel_size=gp.Coordinate((2, 1)), interpolatable=True)

        hdf5_source = gp.Hdf5Source(self.fake_data_file, {key: 'testdata'}, array_specs={key: spec})

        request = gp.BatchRequest()
        shape = gp.Coordinate((3, 3))
        request.add(key, shape, voxel_size=gp.Coordinate((3, 1)))

        shift_node = gp.ShiftAugment(prob_slip=0.2, prob_shift=0.2, sigma=1, shift_axis=0)
        with gp.build((hdf5_source + shift_node)) as b:
            with self.assertRaises(AssertionError):
                b.request_batch(request)

    def test_pipeline2(self):

        key = gp.ArrayKey("TEST_ARRAY")
        spec = gp.ArraySpec(voxel_size=gp.Coordinate((3, 1)), interpolatable=True)

        hdf5_source = gp.Hdf5Source(self.fake_data_file, {key: 'testdata'}, array_specs={key: spec})

        request = gp.BatchRequest()
        shape = gp.Coordinate((3, 3))
        request.add(key, shape, voxel_size=gp.Coordinate((3, 1)))

        shift_node = gp.ShiftAugment(prob_slip=0.2, prob_shift=0.2, sigma=1, shift_axis=0)
        with gp.build((hdf5_source + shift_node)) as b:
            b.request_batch(request)

    def test_pipeline3(self):
        array_key = gp.ArrayKey("TEST_ARRAY")
        points_key = gp.PointsKey("TEST_POINTS")
        voxel_size = gp.Coordinate((1, 1))
        spec = gp.ArraySpec(voxel_size=voxel_size, interpolatable=True)

        hdf5_source = gp.Hdf5Source(self.fake_data_file, {array_key: 'testdata'}, array_specs={array_key: spec})
        csv_source = gp.CsvPointsSource(self.fake_points_file,
                                        points_key,
                                        gp.PointsSpec(roi=gp.Roi(shape=gp.Coordinate((100, 100)), offset=(0, 0))))

        request = gp.BatchRequest()
        shape = gp.Coordinate((60, 60))
        request.add(array_key, shape, voxel_size=gp.Coordinate((1, 1)))
        request.add(points_key, shape)

        shift_node = gp.ShiftAugment(prob_slip=0.2, prob_shift=0.2, sigma=5, shift_axis=0)
        pipeline = ((hdf5_source, csv_source)
                    + gp.MergeProvider()
                    + gp.RandomLocation(ensure_nonempty=points_key)
                    + shift_node)
        with gp.build(pipeline) as b:
            request = b.request_batch(request)
            # print(request[points_key])

        target_vals = [self.fake_data[point[0]][point[1]] for point in self.fake_points]
        result_data = request[array_key].data
        result_points = request[points_key].data
        result_vals = [result_data[int(point.location[0])][int(point.location[1])] for point in result_points.values()]

        for result_val in result_vals:
            self.assertTrue(result_val in target_vals,
                            msg="result value {} at points {} not in target values {} at points {}".format(
                                result_val, list(result_points.values()), target_vals, self.fake_points))

    ##################
    # shift_and_crop #
    ##################

    def test_shift_and_crop_static(self):
        shift_node = gp.ShiftAugment(sigma=1, shift_axis=0)
        shift_node.ndim = 2
        upstream_arr = np.arange(16).reshape(4, 4)
        sub_shift_array = np.zeros(8, dtype=int).reshape(4, 2)
        roi_shape = (4, 4)
        voxel_size = gp.Coordinate((1, 1))

        downstream_arr = np.arange(16).reshape(4, 4)

        result = shift_node.shift_and_crop(upstream_arr, roi_shape, sub_shift_array, voxel_size)
        self.assertTrue(np.array_equal(result, downstream_arr))

    def test_shift_and_crop1(self):
        shift_node = gp.ShiftAugment(sigma=1, shift_axis=0)
        shift_node.ndim = 2
        upstream_arr = np.arange(16).reshape(4, 4)
        sub_shift_array = np.zeros(8, dtype=int).reshape(4, 2)
        sub_shift_array[:, 1] = np.array([0, -1, 1, 0], dtype=int)
        roi_shape = (4, 2)
        voxel_size = gp.Coordinate((1, 1))

        downstream_arr = np.array([[1, 2],
                                   [6, 7],
                                   [8, 9],
                                   [13, 14]], dtype=int)

        result = shift_node.shift_and_crop(upstream_arr, roi_shape, sub_shift_array, voxel_size)
        self.assertTrue(np.array_equal(result, downstream_arr))

    def test_shift_and_crop2(self):
        shift_node = gp.ShiftAugment(sigma=1, shift_axis=0)
        shift_node.ndim = 2
        upstream_arr = np.arange(16).reshape(4, 4)
        sub_shift_array = np.zeros(8, dtype=int).reshape(4, 2)
        sub_shift_array[:, 1] = np.array([0, -1, -2, 0], dtype=int)
        roi_shape = (4, 2)
        voxel_size = gp.Coordinate((1, 1))

        downstream_arr = np.array([[0, 1],
                                   [5, 6],
                                   [10, 11],
                                   [12, 13]], dtype=int)

        result = shift_node.shift_and_crop(upstream_arr, roi_shape, sub_shift_array, voxel_size)
        self.assertTrue(np.array_equal(result, downstream_arr))

    def test_shift_and_crop3(self):
        shift_node = gp.ShiftAugment(sigma=1, shift_axis=1)
        shift_node.ndim = 2
        upstream_arr = np.arange(16).reshape(4, 4)
        sub_shift_array = np.zeros(8, dtype=int).reshape(4, 2)
        sub_shift_array[:, 0] = np.array([0, 1, 0, 2], dtype=int)
        roi_shape = (2, 4)
        voxel_size = gp.Coordinate((1, 1))

        downstream_arr = np.array([[8, 5, 10, 3],
                                   [12, 9, 14, 7]], dtype=int)

        result = shift_node.shift_and_crop(upstream_arr, roi_shape, sub_shift_array, voxel_size)
        # print(result)
        self.assertTrue(np.array_equal(result, downstream_arr))

    def test_shift_and_crop4(self):
        shift_node = gp.ShiftAugment(sigma=1, shift_axis=1)
        shift_node.ndim = 2
        upstream_arr = np.arange(16).reshape(4, 4)
        sub_shift_array = np.zeros(8, dtype=int).reshape(4, 2)
        sub_shift_array[:, 0] = np.array([0, 2, 0, 4], dtype=int)
        roi_shape = (4, 4)
        voxel_size = gp.Coordinate((2, 1))

        downstream_arr = np.array([[8, 5, 10, 3],
                                   [12, 9, 14, 7]], dtype=int)

        result = shift_node.shift_and_crop(upstream_arr, roi_shape, sub_shift_array, voxel_size)
        # print(result)
        self.assertTrue(np.array_equal(result, downstream_arr))

        result = shift_node.shift_and_crop(upstream_arr, roi_shape, sub_shift_array, voxel_size)
        # print(result)
        self.assertTrue(np.array_equal(result, downstream_arr))

    ##################
    # shift_points   #
    ##################

    @staticmethod
    def points_equal(points1, points2):
        keys1 = set(points1.keys())
        keys2 = set(points2.keys())
        if keys1 != keys2:
            return False
        for key in keys1:
            p1 = points1[key].location
            p2 = points2[key].location
            if not np.array_equal(p1, p2):
                return False
        return True

    def test_points_equal(self):
        points1 = {1: gp.Point([0, 1])}
        points2 = {1: gp.Point([0, 1])}
        self.assertTrue(self.points_equal(points1, points2))

        points1[2] = gp.Point([1, 2])
        points2[2] = gp.Point([2, 1])
        self.assertFalse(self.points_equal(points1, points2))

    def test_shift_points1(self):
        data = {1: gp.Point([0, 1])}
        spec = gp.PointsSpec(gp.Roi(offset=(0, 0), shape=(5, 5)))
        points = gp.Points(data, spec)
        request_roi = gp.Roi(offset=(0, 1), shape=(5, 3))
        shift_array = np.array([[0, -1],
                                [0, -1],
                                [0, 0],
                                [0, 0],
                                [0, 1]], dtype=int)
        lcm_voxel_size = gp.Coordinate((1, 1))

        shifted_points = gp.Points({}, gp.PointsSpec(request_roi))
        result = gp.ShiftAugment.shift_points(points,
                                        request_roi,
                                        shift_array,
                                        shift_axis=0,
                                        lcm_voxel_size=lcm_voxel_size)
        # print(result)
        self.assertTrue(self.points_equal(result.data, shifted_points.data))
        self.assertTrue(result.spec == gp.PointsSpec(request_roi))

    def test_shift_points2(self):
        data = {1: gp.Point([0, 1])}
        spec = gp.PointsSpec(gp.Roi(offset=(0, 0), shape=(5, 5)))
        points = gp.Points(data, spec)
        request_roi = gp.Roi(offset=(0, 1), shape=(5, 3))
        shift_array = np.array([[0, 0],
                                [0, -1],
                                [0, 0],
                                [0, 0],
                                [0, 1]], dtype=int)
        lcm_voxel_size = gp.Coordinate((1, 1))

        result = gp.ShiftAugment.shift_points(points, request_roi, shift_array, shift_axis=0, lcm_voxel_size=lcm_voxel_size)
        # print("test 2", result.data, data)
        self.assertTrue(self.points_equal(result.data, data))
        self.assertTrue(result.spec == gp.PointsSpec(request_roi))

    def test_shift_points3(self):
        data = {1: gp.Point([0, 1])}
        spec = gp.PointsSpec(gp.Roi(offset=(0, 0), shape=(5, 5)))
        points = gp.Points(data, spec)
        request_roi = gp.Roi(offset=(0, 1), shape=(5, 3))
        shift_array = np.array([[0, 1],
                                [0, -1],
                                [0, 0],
                                [0, 0],
                                [0, 1]], dtype=int)
        lcm_voxel_size = gp.Coordinate((1, 1))

        shifted_points = gp.Points({1: gp.Point([0, 2])}, gp.PointsSpec(request_roi))
        result = gp.ShiftAugment.shift_points(points,
                                        request_roi,
                                        shift_array,
                                        shift_axis=0,
                                        lcm_voxel_size=lcm_voxel_size)
        # print("test 3", result.data, shifted_points.data)
        self.assertTrue(self.points_equal(result.data, shifted_points.data))
        self.assertTrue(result.spec == gp.PointsSpec(request_roi))

    def test_shift_points4(self):
        data = {0: gp.Point([1, 0]),
                1: gp.Point([1, 1]),
                2: gp.Point([1, 2]),
                3: gp.Point([1, 3]),
                4: gp.Point([1, 4])}
        spec = gp.PointsSpec(gp.Roi(offset=(0, 0), shape=(5, 5)))
        points = gp.Points(data, spec)
        request_roi = gp.Roi(offset=(1, 0), shape=(3, 5))
        shift_array = np.array([[1, 0],
                                [-1, 0],
                                [0, 0],
                                [-1, 0],
                                [1, 0]], dtype=int)

        lcm_voxel_size = gp.Coordinate((1, 1))
        shifted_data = {0: gp.Point([2, 0]),
                        2: gp.Point([1, 2]),
                        4: gp.Point([2, 4])}
        result = gp.ShiftAugment.shift_points(points,
                                        request_roi,
                                        shift_array,
                                        shift_axis=1,
                                        lcm_voxel_size=lcm_voxel_size)
        # print("test 4", result.data, shifted_data)
        self.assertTrue(self.points_equal(result.data, shifted_data))
        self.assertTrue(result.spec == gp.PointsSpec(request_roi))

    def test_shift_points5(self):
        data = {0: gp.Point([3, 0]),
                1: gp.Point([3, 2]),
                2: gp.Point([3, 4]),
                3: gp.Point([3, 6]),
                4: gp.Point([3, 8])}
        spec = gp.PointsSpec(gp.Roi(offset=(0, 0), shape=(15, 10)))
        points = gp.Points(data, spec)
        request_roi = gp.Roi(offset=(3, 0), shape=(9, 10))
        shift_array = np.array([[3, 0],
                                [-3, 0],
                                [0, 0],
                                [-3, 0],
                                [3, 0]], dtype=int)

        lcm_voxel_size = gp.Coordinate((3, 2))
        shifted_data = {0: gp.Point([6, 0]),
                        2: gp.Point([3, 4]),
                        4: gp.Point([6, 8])}
        result = gp.ShiftAugment.shift_points(points, request_roi, shift_array, shift_axis=1, lcm_voxel_size=lcm_voxel_size)
        # print("test 4", result.data, shifted_data)
        self.assertTrue(self.points_equal(result.data, shifted_data))
        self.assertTrue(result.spec == gp.PointsSpec(request_roi))

    #######################
    # get_sub_shift_array #
    #######################

    def test_get_sub_shift_array1(self):
        total_roi = gp.Roi(offset=(0, 0), shape=(6, 6))
        item_roi = gp.Roi(offset=(1, 2), shape=(3, 3))
        shift_array = np.arange(12).reshape(6, 2).astype(int)
        shift_axis = 1
        lcm_voxel_size = gp.Coordinate((1, 1))

        sub_shift_array = np.array([[4, 5],
                                    [6, 7],
                                    [8, 9]], dtype=int)
        result = gp.ShiftAugment.get_sub_shift_array(total_roi,
                                               item_roi,
                                               shift_array,
                                               shift_axis,
                                               lcm_voxel_size)
        # print(result)
        self.assertTrue(np.array_equal(result, sub_shift_array))

    def test_get_sub_shift_array2(self):
        total_roi = gp.Roi(offset=(0, 0), shape=(6, 6))
        item_roi = gp.Roi(offset=(1, 2), shape=(3, 3))
        shift_array = np.arange(12).reshape(6, 2).astype(int)
        shift_axis = 0
        lcm_voxel_size = gp.Coordinate((1, 1))

        sub_shift_array = np.array([[2, 3],
                                    [4, 5],
                                    [6, 7]], dtype=int)
        result = gp.ShiftAugment.get_sub_shift_array(total_roi,
                                               item_roi,
                                               shift_array,
                                               shift_axis,
                                               lcm_voxel_size)
        self.assertTrue(np.array_equal(result, sub_shift_array))

    def test_get_sub_shift_array3(self):
        total_roi = gp.Roi(offset=(0, 0), shape=(18, 12))
        item_roi = gp.Roi(offset=(3, 4), shape=(9, 6))
        shift_array = np.arange(12).reshape(6, 2).astype(int)
        shift_axis = 0
        lcm_voxel_size = gp.Coordinate((3, 2))

        sub_shift_array = np.array([[2, 3],
                                    [4, 5],
                                    [6, 7]], dtype=int)
        result = gp.ShiftAugment.get_sub_shift_array(total_roi,
                                               item_roi,
                                               shift_array,
                                               shift_axis,
                                               lcm_voxel_size)
        # print(result)
        self.assertTrue(np.array_equal(result, sub_shift_array))

    ################################
    # construct_global_shift_array #
    ################################

    def test_construct_global_shift_array_static(self):
        shift_axis_len = 5
        shift_sigmas = (0.0, 1.0)
        prob_slip = 0
        prob_shift = 0
        lcm_voxel_size = gp.Coordinate((1, 1))

        shift_array = np.zeros(shape=(shift_axis_len, len(shift_sigmas)), dtype=int)
        result = gp.ShiftAugment.construct_global_shift_array(shift_axis_len,
                                                        shift_sigmas,
                                                        prob_shift,
                                                        prob_slip,
                                                        lcm_voxel_size)
        self.assertTrue(np.array_equal(result, shift_array))

    def test_construct_global_shift_array1(self):
        shift_axis_len = 5
        shift_sigmas = (0.0, 1.0)
        prob_slip = 1
        prob_shift = 0
        lcm_voxel_size = gp.Coordinate((1, 1))

        shift_array = np.array([[0, 0],
                                [0, -1],
                                [0, 1],
                                [0, 0],
                                [0, 1]], dtype=int)
        result = gp.ShiftAugment.construct_global_shift_array(shift_axis_len,
                                                        shift_sigmas,
                                                        prob_slip,
                                                        prob_shift,
                                                        lcm_voxel_size)
        # print(result)
        self.assertTrue(len(result) == shift_axis_len)
        for position_shift in result:
            self.assertTrue(position_shift[0] == 0)
        self.assertTrue(np.array_equal(shift_array, result))

    def test_construct_global_shift_array2(self):
        shift_axis_len = 5
        shift_sigmas = (0.0, 1.0)
        prob_slip = 0
        prob_shift = 1
        lcm_voxel_size = gp.Coordinate((1, 1))

        shift_array = np.array([[0, 0],
                                [0, -1],
                                [0, 0],
                                [0, 0],
                                [0, 1]], dtype=int)
        result = gp.ShiftAugment.construct_global_shift_array(shift_axis_len,
                                                        shift_sigmas,
                                                        prob_slip,
                                                        prob_shift,
                                                        lcm_voxel_size)
        self.assertTrue(len(result) == shift_axis_len)
        for position_shift in result:
            self.assertTrue(position_shift[0] == 0)
        self.assertTrue(np.array_equal(shift_array, result))

    def test_construct_global_shift_array3(self):
        shift_axis_len = 5
        shift_sigmas = (0.0, 4.0)
        prob_slip = 0
        prob_shift = 1
        lcm_voxel_size = gp.Coordinate((1, 3))

        shift_array = np.array([[0, 3],
                                [0, 0],
                                [0, 6],
                                [0, 6],
                                [0, 12]], dtype=int)
        result = gp.ShiftAugment.construct_global_shift_array(shift_axis_len,
                                                        shift_sigmas,
                                                        prob_slip,
                                                        prob_shift,
                                                        lcm_voxel_size)
        # print(result)
        self.assertTrue(len(result) == shift_axis_len)
        for position_shift in result:
            self.assertTrue(position_shift[0] == 0)
        self.assertTrue(np.array_equal(shift_array, result))

    ########################
    # compute_upstream_roi #
    ########################

    def test_compute_upstream_roi_static(self):
        request_roi = gp.Roi(offset=(0, 0), shape=(5, 10))
        sub_shift_array = np.array([[0, 0],
                                    [0, 0],
                                    [0, 0],
                                    [0, 0],
                                    [0, 0]], dtype=int)

        upstream_roi = gp.Roi(offset=(0, 0), shape=(5, 10))
        result = gp.ShiftAugment.compute_upstream_roi(request_roi, sub_shift_array)
        self.assertTrue(upstream_roi == result)

    def test_compute_upstream_roi1(self):
        request_roi = gp.Roi(offset=(0, 0), shape=(5, 10))
        sub_shift_array = np.array([[0, 0],
                                    [0, -1],
                                    [0, 0],
                                    [0, 0],
                                    [0, 1]], dtype=int)

        upstream_roi = gp.Roi(offset=(0, -1), shape=(5, 12))
        result = gp.ShiftAugment.compute_upstream_roi(request_roi, sub_shift_array)
        self.assertTrue(upstream_roi == result)

    def test_compute_upstream_roi2(self):
        request_roi = gp.Roi(offset=(0, 0), shape=(5, 10))
        sub_shift_array = np.array([[2, 0],
                                    [-1, 0],
                                    [5, 0],
                                    [-2, 0],
                                    [0, 0]], dtype=int)

        upstream_roi = gp.Roi(offset=(-5, 0), shape=(12, 10))
        result = gp.ShiftAugment.compute_upstream_roi(request_roi, sub_shift_array)
        self.assertTrue(upstream_roi == result)


if __name__ == '__main__':
    unittest.main()
