import unittest
import numpy as np
import random
from gunpowder import Jitter, Roi, Points, Point, PointsSpec


class TestJitter2D(unittest.TestCase):

    def setUp(self):
        random.seed(12345)
        np.random.seed(12345)

    # def test_prepare(self):
    #     request = BatchRequest()
    #     key = ArrayKey("TEST_ARRAY")
    #     shape = (3, 3)
    #     request.add(key, shape)
    #
    #     jitter_node = Jitter(sigma=1, jitter_axis=0)
    #
    #     jitter_node.prepare(request)
    #     self.assertTrue(jitter_node.ndim == 2)
    #     self.assertTrue(jitter_node.jitter_sigmas == [0.0, 1.0])
    #     self.assertTrue(jitter_node.shift_axes == (1,))

    ##################
    # shift_and_crop #
    ##################

    def test_shift_and_crop_static(self):
        jitter_node = Jitter(sigma=1, jitter_axis=0)
        jitter_node.ndim = 2
        upstream_arr = np.arange(16).reshape(4, 4)
        sub_shift_array = np.zeros(8, dtype=int).reshape(4, 2)
        roi_shape = (4, 4)

        downstream_arr = np.arange(16).reshape(4, 4)

        result = jitter_node.shift_and_crop(upstream_arr, roi_shape, sub_shift_array)
        self.assertTrue(np.array_equal(result, downstream_arr))

    def test_shift_and_crop1(self):
        jitter_node = Jitter(sigma=1, jitter_axis=0)
        jitter_node.ndim = 2
        upstream_arr = np.arange(16).reshape(4, 4)
        sub_shift_array = np.zeros(8, dtype=int).reshape(4, 2)
        sub_shift_array[:, 1] = np.array([0, -1, 1, 0], dtype=int)
        roi_shape = (4, 2)

        downstream_arr = np.array([[1, 2],
                                   [6, 7],
                                   [8, 9],
                                   [13, 14]], dtype=int)

        result = jitter_node.shift_and_crop(upstream_arr, roi_shape, sub_shift_array)
        self.assertTrue(np.array_equal(result, downstream_arr))

    def test_shift_and_crop2(self):
        jitter_node = Jitter(sigma=1, jitter_axis=0)
        jitter_node.ndim = 2
        upstream_arr = np.arange(16).reshape(4, 4)
        sub_shift_array = np.zeros(8, dtype=int).reshape(4, 2)
        sub_shift_array[:, 1] = np.array([0, -1, -2, 0], dtype=int)
        roi_shape = (4, 2)

        downstream_arr = np.array([[0, 1],
                                   [5, 6],
                                   [10, 11],
                                   [12, 13]], dtype=int)

        result = jitter_node.shift_and_crop(upstream_arr, roi_shape, sub_shift_array)
        self.assertTrue(np.array_equal(result, downstream_arr))

    def test_shift_and_crop3(self):
        jitter_node = Jitter(sigma=1, jitter_axis=1)
        jitter_node.ndim = 2
        upstream_arr = np.arange(16).reshape(4, 4)
        sub_shift_array = np.zeros(8, dtype=int).reshape(4, 2)
        sub_shift_array[:, 0] = np.array([0, 1, 0, 2], dtype=int)
        roi_shape = (2, 4)

        downstream_arr = np.array([[8, 5, 10, 3],
                                   [12, 9, 14, 7]], dtype=int)

        result = jitter_node.shift_and_crop(upstream_arr, roi_shape, sub_shift_array)
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
        points1 = {1: Point([0, 1])}
        points2 = {1: Point([0, 1])}
        self.assertTrue(self.points_equal(points1, points2))

        points1[2] = Point([1, 2])
        points2[2] = Point([2, 1])
        self.assertFalse(self.points_equal(points1, points2))

    def test_shift_points1(self):
        data = {1: Point([0, 1])}
        spec = PointsSpec(Roi(offset=(0, 0), shape=(5, 5)))
        points = Points(data, spec)
        request_roi = Roi(offset=(0, 1), shape=(5, 3))
        shift_array = np.array([[0, -1],
                                [0, -1],
                                [0, 0],
                                [0, 0],
                                [0, 1]], dtype=int)

        shifted_points = Points({}, PointsSpec(request_roi))
        result = Jitter.shift_points(points, request_roi, shift_array, jitter_axis=0)
        # print(result)
        self.assertTrue(self.points_equal(result.data, shifted_points.data))
        self.assertTrue(result.spec == PointsSpec(request_roi))

    def test_shift_points2(self):
        data = {1: Point([0, 1])}
        spec = PointsSpec(Roi(offset=(0, 0), shape=(5, 5)))
        points = Points(data, spec)
        request_roi = Roi(offset=(0, 1), shape=(5, 3))
        shift_array = np.array([[0, 0],
                                [0, -1],
                                [0, 0],
                                [0, 0],
                                [0, 1]], dtype=int)

        result = Jitter.shift_points(points, request_roi, shift_array, jitter_axis=0)
        # print("test 2", result.data, data)
        self.assertTrue(self.points_equal(result.data, data))
        self.assertTrue(result.spec == PointsSpec(request_roi))

    def test_shift_points3(self):
        data = {1: Point([0, 1])}
        spec = PointsSpec(Roi(offset=(0, 0), shape=(5, 5)))
        points = Points(data, spec)
        request_roi = Roi(offset=(0, 1), shape=(5, 3))
        shift_array = np.array([[0, 1],
                                [0, -1],
                                [0, 0],
                                [0, 0],
                                [0, 1]], dtype=int)

        shifted_points = Points({1: Point([0, 2])}, PointsSpec(request_roi))
        result = Jitter.shift_points(points, request_roi, shift_array, jitter_axis=0)
        # print("test 3", result.data, shifted_points.data)
        self.assertTrue(self.points_equal(result.data, shifted_points.data))
        self.assertTrue(result.spec == PointsSpec(request_roi))

    def test_shift_points4(self):
        data = {0: Point([1, 0]),
                1: Point([1, 1]),
                2: Point([1, 2]),
                3: Point([1, 3]),
                4: Point([1, 4])}
        spec = PointsSpec(Roi(offset=(0, 0), shape=(5, 5)))
        points = Points(data, spec)
        request_roi = Roi(offset=(1, 0), shape=(3, 5))
        shift_array = np.array([[1, 0],
                                [-1, 0],
                                [0, 0],
                                [-1, 0],
                                [1, 0]], dtype=int)
        shifted_data = {0: Point([2, 0]),
                        2: Point([1, 2]),
                        4: Point([2, 4])}
        result = Jitter.shift_points(points, request_roi, shift_array, jitter_axis=1)
        # print("test 4", result.data, shifted_data)
        self.assertTrue(self.points_equal(result.data, shifted_data))
        self.assertTrue(result.spec == PointsSpec(request_roi))

    #######################
    # get_sub_shift_array #
    #######################

    def test_get_sub_shift_array1(self):
        total_roi = Roi(offset=(0, 0), shape=(6, 6))
        item_roi = Roi(offset=(1, 2), shape=(3, 3))
        shift_array = np.arange(12).reshape(6, 2).astype(int)
        jitter_axis = 1

        sub_shift_array = np.array([[4, 5],
                                    [6, 7],
                                    [8, 9]], dtype=int)
        result = Jitter.get_sub_shift_array(total_roi, item_roi, shift_array, jitter_axis)
        # print(result)
        self.assertTrue(np.array_equal(result, sub_shift_array))

    def test_get_sub_shift_array2(self):
        total_roi = Roi(offset=(0, 0), shape=(6, 6))
        item_roi = Roi(offset=(1, 2), shape=(3, 3))
        shift_array = np.arange(12).reshape(6, 2).astype(int)
        jitter_axis = 0

        sub_shift_array = np.array([[2, 3],
                                    [4, 5],
                                    [6, 7]], dtype=int)
        result = Jitter.get_sub_shift_array(total_roi, item_roi, shift_array, jitter_axis)
        self.assertTrue(np.array_equal(result, sub_shift_array))

    ################################
    # construct_global_shift_array #
    ################################

    def test_construct_global_shift_array_static(self):
        jitter_axis_len = 5
        jitter_sigmas = (0.0, 1.0)
        prob_slip = 0
        prob_shift = 0

        shift_array = np.zeros(shape=(jitter_axis_len, len(jitter_sigmas)), dtype=int)
        result = Jitter.construct_global_shift_array(jitter_axis_len, jitter_sigmas, prob_shift, prob_slip)
        self.assertTrue(np.array_equal(result, shift_array))

    def test_construct_global_shift_array1(self):
        jitter_axis_len = 5
        jitter_sigmas = (0.0, 1.0)
        prob_slip = 1
        prob_shift = 0

        shift_array = np.array([[0, 0],
                                [0, -1],
                                [0, 1],
                                [0, 0],
                                [0, 1]], dtype=int)
        result = Jitter.construct_global_shift_array(jitter_axis_len, jitter_sigmas, prob_slip, prob_shift)
        # print(result)
        self.assertTrue(len(result) == jitter_axis_len)
        for position_shift in result:
            self.assertTrue(position_shift[0] == 0)
        self.assertTrue(np.array_equal(shift_array, result))

    def test_construct_global_shift_array2(self):
        jitter_axis_len = 5
        jitter_sigmas = (0.0, 1.0)
        prob_slip = 0
        prob_shift = 1
        shift_array = np.array([[0, 0],
                                [0, -1],
                                [0, 0],
                                [0, 0],
                                [0, 1]], dtype=int)
        result = Jitter.construct_global_shift_array(jitter_axis_len, jitter_sigmas, prob_slip, prob_shift)
        self.assertTrue(len(result) == jitter_axis_len)
        for position_shift in result:
            self.assertTrue(position_shift[0] == 0)
        self.assertTrue(np.array_equal(shift_array, result))

    ########################
    # compute_upstream_roi #
    ########################

    def test_compute_upstream_roi_static(self):
        request_roi = Roi(offset=(0, 0), shape=(5, 10))
        sub_shift_array = np.array([[0, 0],
                                    [0, 0],
                                    [0, 0],
                                    [0, 0],
                                    [0, 0]], dtype=int)

        upstream_roi = Roi(offset=(0, 0), shape=(5, 10))
        result = Jitter.compute_upstream_roi(request_roi, sub_shift_array)
        self.assertTrue(upstream_roi == result)

    def test_compute_upstream_roi1(self):
        request_roi = Roi(offset=(0, 0), shape=(5, 10))
        sub_shift_array = np.array([[0, 0],
                                    [0, -1],
                                    [0, 0],
                                    [0, 0],
                                    [0, 1]], dtype=int)

        upstream_roi = Roi(offset=(0, -1), shape=(5, 12))
        result = Jitter.compute_upstream_roi(request_roi, sub_shift_array)
        self.assertTrue(upstream_roi == result)

    def test_compute_upstream_roi2(self):
        request_roi = Roi(offset=(0, 0), shape=(5, 10))
        sub_shift_array = np.array([[2, 0],
                                    [-1, 0],
                                    [5, 0],
                                    [-2, 0],
                                    [0, 0]], dtype=int)

        upstream_roi = Roi(offset=(-5, 0), shape=(12, 10))
        result = Jitter.compute_upstream_roi(request_roi, sub_shift_array)
        self.assertTrue(upstream_roi == result)


if __name__ == '__main__':
    unittest.main()
