import unittest
from gunpowder import *
from gunpowder.points import PointsKeys, Points, Point
from .provider_test import ProviderTest

import numpy as np
import math
from random import randint

class PointTestSource3D(BatchProvider):

    def setup(self):

        self.points = {
            0: Point([0, 0, 0]),
            1: Point([0, 10, 0]),
            2: Point([0, 20, 0]),
            3: Point([0, 30, 0]),
            4: Point([0, 40, 0]),
            5: Point([0, 50, 0]),
        }

        self.provides(
            PointsKeys.TEST_POINTS,
            PointsSpec(
                roi=Roi((-100, -100, -100), (200, 200, 200))
            ))

        self.provides(
            ArrayKeys.TEST_LABELS,
            ArraySpec(
                roi=Roi((-100, -100, -100), (200, 200, 200)),
                voxel_size=Coordinate((4, 1, 1)),
                interpolatable=False
            ))

    def point_to_voxel(self, array_roi, location):

        # location is in world units, get it into voxels
        location = location/self.spec[ArrayKeys.TEST_LABELS].voxel_size

        # shift location relative to beginning of array roi
        location -= array_roi.get_begin()/self.spec[ArrayKeys.TEST_LABELS].voxel_size

        return tuple(
            slice(int(l-2), int(l+3))
            for l in location)

    def provide(self, request):

        batch = Batch()

        roi_points = request[PointsKeys.TEST_POINTS].roi
        roi_array = request[ArrayKeys.TEST_LABELS].roi
        roi_voxel = roi_array//self.spec[ArrayKeys.TEST_LABELS].voxel_size

        data = np.zeros(roi_voxel.get_shape(), dtype=np.uint32)
        data[:,::2] = 100

        for i, point in self.points.items():
            loc = self.point_to_voxel(roi_array, point.location)
            data[loc] = i

        spec = self.spec[ArrayKeys.TEST_LABELS].copy()
        spec.roi = roi_array
        batch.arrays[ArrayKeys.TEST_LABELS] = Array(
            data,
            spec=spec)

        points = {}
        for i, point in self.points.items():
            if roi_points.contains(point.location):
                points[i] = point
        batch.points[PointsKeys.TEST_POINTS] = Points(
            points,
            PointsSpec(roi=roi_points))

        return batch


class TestElasticAugment(ProviderTest):

    def test_3d_basics(self):

        test_labels = ArrayKey('TEST_LABELS')
        test_points = PointsKey('TEST_POINTS')
        test_raster = ArrayKey('TEST_RASTER')

        pipeline = (

            PointTestSource3D() +
            ElasticAugment(
                [10, 10, 10],
                [0.1, 0.1, 0.1],
                # [0, 0, 0], # no jitter
                [0, 2.0*math.pi]) + # rotate randomly
                # [math.pi/4, math.pi/4]) + # rotate by 45 deg
                # [0, 0]) + # no rotation
            RasterizePoints(
                test_points,
                test_raster,
                settings=RasterizationSettings(
                    radius=2,
                    mode='peak')) +
            Snapshot(
                {
                    test_labels: 'volumes/labels',
                    test_raster: 'volumes/raster'
                },
                dataset_dtypes={test_raster: np.float32},
                output_dir=self.path_to(),
                output_filename='elastic_augment_test{id}-{iteration}.hdf'
            )
        )

        for _ in range(5):

            with build(pipeline):

                request_roi = Roi(
                    (-20, -20, -20),
                    (40, 40, 40))

                request = BatchRequest()
                request[test_labels] = ArraySpec(roi=request_roi)
                # request[test_points] = PointsSpec(roi=request_roi)
                request[test_raster] = ArraySpec(roi=request_roi)

                batch = pipeline.request_batch(request)
                labels = batch.arrays[test_labels]
                points = batch.points[test_points]

                # the point at (0, 0, 0) should not have moved
                self.assertTrue(0 in points.data)

                labels_data_roi = (
                    labels.spec.roi -
                    labels.spec.roi.get_begin())/labels.spec.voxel_size

                # points should have moved together with the voxels
                for i, point in points.data.items():
                    loc = point.location - labels.spec.roi.get_begin()
                    loc = loc/labels.spec.voxel_size
                    loc = Coordinate(int(round(x)) for x in loc)
                    if labels_data_roi.contains(loc):
                        self.assertEqual(labels.data[loc], i)
