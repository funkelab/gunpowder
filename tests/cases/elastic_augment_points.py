import unittest
from gunpowder import *
from gunpowder.points import PointsKeys, Points, Point

import numpy as np
import math
from random import randint

class PointTestSource3D(BatchProvider):

    def __init__(self, voxel_size, object_location, point_dic):
        self.voxel_size = voxel_size
        self.object_location = object_location
        self.point_dic = point_dic

    def setup(self):

        self.provides(
            PointsKeys.PRESYN,
            PointsSpec(roi=Roi((-100, -100, -100), (200, 200, 200))))
        self.provides(
            ArrayKeys.GT_LABELS,
            ArraySpec(
                roi=Roi((-100, -100, -100), (200, 200, 200)),
                voxel_size=self.voxel_size))

    def provide(self, request):
        batch = Batch()
        roi_points = request[PointsKeys.PRESYN].roi
        roi_array = request[ArrayKeys.GT_LABELS].roi
        image = np.zeros(roi_array.get_shape()/self.voxel_size)
        image[self.object_location] = 1

        id_to_point = {}
        for point_id, location in self.point_dic.items():
            location += roi_points.get_offset()
            if roi_points.contains(location):
                id_to_point[point_id] = Point(location)

        batch.points[PointsKeys.PRESYN] = Points(
            id_to_point,
            PointsSpec(roi=roi_points))
        spec = self.spec[ArrayKeys.GT_LABELS].copy()
        spec.roi = roi_array
        batch.arrays[ArrayKeys.GT_LABELS] = Array(
            image,
            spec=spec)
        return batch


class TestElasticAugment(unittest.TestCase):

    def test_3d_basics(self):
        # Check correct transformation of points for 5 random elastic augmentations. The correct transformation is
        # tested by also augmenting a array with a specific object/region labeled. The point to test is placed
        # within the object. Augmenting the array with the object together with the point should result in a
        # transformed array in which the point is still located within the object.
        voxel_size = Coordinate((2, 1, 1))

        PointsKey('PRESYN')

        for i in range(5):
            object_location = tuple([slice(30/voxel_size[0], 40/voxel_size[0]),
                                     slice(30/voxel_size[1], 40/voxel_size[1]),
                                     slice(30/voxel_size[2], 40/voxel_size[2])])
            points_to_test = {}
            points_to_test[0] = np.array((35, 35, 35))  # point inside object
            points_to_test[2] = np.array((20, 20, 20))  # point outside object
            points_to_test[5] = np.array((35, 35, 35)) # point with different id but same location
            points_to_test[10] = np.array((150, 150, 150)) # point should disappear because outside of roi

            # Random elastic augmentation hyperparameter
            subsample = randint(1, 4)
            control_point_spacing = [randint(1, 20), randint(1, 20), randint(1, 2)]

            source_node = PointTestSource3D(voxel_size=voxel_size, object_location=object_location,
                                            point_dic=points_to_test)
            elastic_augm_node = ElasticAugment(control_point_spacing, [0, 2, 2],
                                               [0, math.pi / 2.0], subsample=subsample)
            pipeline = source_node + elastic_augm_node

            with build(pipeline):
                request = BatchRequest()
                window_request = Coordinate((50, 50, 50))

                request.add(PointsKeys.PRESYN, window_request)
                request.add(ArrayKeys.GT_LABELS, window_request)
                batch = pipeline.request_batch(request)

                exp_loc_in_object = batch.points[PointsKeys.PRESYN].data[0].location/voxel_size
                exp_loc_out_object = batch.points[PointsKeys.PRESYN].data[2].location/voxel_size
                array = batch.arrays[ArrayKeys.GT_LABELS].data
                self.assertTrue(array[tuple(exp_loc_in_object)] == 1)
                self.assertTrue(array[tuple(exp_loc_out_object)] == 0)
                self.assertTrue(5 in batch.points[PointsKeys.PRESYN].data)
                self.assertFalse(10 in batch.points[PointsKeys.PRESYN].data)
