from .provider_test import ProviderTest
from gunpowder import *
from gunpowder.points import PointsKeys, Points, Point

import numpy as np
import math
from random import randint

class PointTestSource3D(BatchProvider):

    def __init__(self):

        self.voxel_size = Coordinate((40, 4, 4))

        self.all_points = {
                # corners
                1: Point(Coordinate((-200, -200, -200))),
                2: Point(Coordinate((-200, -200, 199))),
                3: Point(Coordinate((-200, 199, -200))),
                4: Point(Coordinate((-200, 199, 199))),
                5: Point(Coordinate((199, -200, -200))),
                6: Point(Coordinate((199, -200, 199))),
                7: Point(Coordinate((199, 199, -200))),
                8: Point(Coordinate((199, 199, 199))),
                # center
                9: Point(Coordinate((0, 0, 0))),
            }

    def setup(self):

        self.provides(
            PointsKeys.TEST_POINTS,
            PointsSpec(roi=Roi((-100, -100, -100), (300, 300, 300))))

        self.provides(
            ArrayKeys.GT_LABELS,
            ArraySpec(
                roi=Roi((-200, -200, -200), (400, 400, 400)),
                voxel_size=self.voxel_size))

    def provide(self, request):

        batch = Batch()

        roi_points = request[PointsKeys.TEST_POINTS].roi

        # get all points inside the requested ROI
        points = {}
        for point_id, point in self.all_points.items():
            if roi_points.contains(point.location):
                points[point_id] = point

        batch.points[PointsKeys.TEST_POINTS] = Points(
            points,
            PointsSpec(roi=roi_points))

        roi_array = request[ArrayKeys.GT_LABELS].roi

        image = np.ones(roi_array.get_shape()/self.voxel_size, dtype=np.uint64)
        # label half of GT_LABELS differently
        depth = image.shape[0]
        image[0:depth//2] = 2

        spec = self.spec[ArrayKeys.GT_LABELS].copy()
        spec.roi = roi_array
        batch.arrays[ArrayKeys.GT_LABELS] = Array(
            image,
            spec=spec)

        return batch

class TestRasterizePoints(ProviderTest):

    def test_3d(self):

        PointsKey('TEST_POINTS')
        ArrayKey('RASTERIZED')

        pipeline = (
            PointTestSource3D() +
            RasterizePoints(
                PointsKeys.TEST_POINTS,
                ArrayKeys.RASTERIZED,
                ArraySpec(
                    voxel_size=(40, 4, 4)))
        )

        with build(pipeline):

            request = BatchRequest()
            roi = Roi((0, 0, 0), (200, 200, 200))

            request[PointsKeys.TEST_POINTS] = PointsSpec(roi=roi)
            request[ArrayKeys.GT_LABELS] = ArraySpec(roi=roi)
            request[ArrayKeys.RASTERIZED] = ArraySpec(roi=roi)

            batch = pipeline.request_batch(request)

            rasterized = batch.arrays[ArrayKeys.RASTERIZED].data
            self.assertEqual(rasterized[0, 0, 0], 1)
            self.assertEqual(rasterized[2, 20, 20], 0)
            self.assertEqual(rasterized[4, 49, 49], 1)

        # same with different foreground/background labels

        pipeline = (
            PointTestSource3D() +
            RasterizePoints(
                PointsKeys.TEST_POINTS,
                ArrayKeys.RASTERIZED,
                ArraySpec(voxel_size=(40, 4, 4)),
                RasterizationSettings(
                    radius=1,
                    fg_value=0,
                    bg_value=1))
        )

        with build(pipeline):

            request = BatchRequest()
            roi = Roi((0, 0, 0), (200, 200, 200))

            request[PointsKeys.TEST_POINTS] = PointsSpec(roi=roi)
            request[ArrayKeys.GT_LABELS] = ArraySpec(roi=roi)
            request[ArrayKeys.RASTERIZED] = ArraySpec(roi=roi)

            batch = pipeline.request_batch(request)

            rasterized = batch.arrays[ArrayKeys.RASTERIZED].data
            self.assertEqual(rasterized[0, 0, 0], 0)
            self.assertEqual(rasterized[2, 20, 20], 1)
            self.assertEqual(rasterized[4, 49, 49], 0)

        # same with different radius and inner radius

        pipeline = (
            PointTestSource3D() +
            RasterizePoints(
                PointsKeys.TEST_POINTS,
                ArrayKeys.RASTERIZED,
                ArraySpec(voxel_size=(40, 4, 4)),
                RasterizationSettings(
                    radius=40,
                    inner_radius_fraction=0.25,
                    fg_value=1,
                    bg_value=0))
        )

        with build(pipeline):
            request = BatchRequest()
            roi = Roi((0, 0, 0), (200, 200, 200))

            request[PointsKeys.TEST_POINTS] = PointsSpec(roi=roi)
            request[ArrayKeys.GT_LABELS] = ArraySpec(roi=roi)
            request[ArrayKeys.RASTERIZED] = ArraySpec(roi=roi)

            batch = pipeline.request_batch(request)

            rasterized = batch.arrays[ArrayKeys.RASTERIZED].data

            # in the middle of the ball, there should be 0 (since inner radius is set)
            self.assertEqual(rasterized[0, 0, 0], 0)
            # check larger radius: rasterized point (0, 0, 0) should extend in
            # x,y by 10; z, by 1
            self.assertEqual(rasterized[0, 10, 0], 1)
            self.assertEqual(rasterized[0, 0, 10], 1)
            self.assertEqual(rasterized[1, 0, 0], 1)

            self.assertEqual(rasterized[2, 20, 20], 0)
            self.assertEqual(rasterized[4, 49, 49], 0)

        # same with anisotropic radius

        pipeline = (
            PointTestSource3D() +
            RasterizePoints(
                PointsKeys.TEST_POINTS,
                ArrayKeys.RASTERIZED,
                ArraySpec(voxel_size=(40, 4, 4)),
                RasterizationSettings(
                    radius=(40, 40, 20),
                    fg_value=1,
                    bg_value=0))
        )

        with build(pipeline):
            request = BatchRequest()
            roi = Roi((0, 0, 0), (120, 80, 80))

            request[PointsKeys.TEST_POINTS] = PointsSpec(roi=roi)
            request[ArrayKeys.GT_LABELS] = ArraySpec(roi=roi)
            request[ArrayKeys.RASTERIZED] = ArraySpec(roi=roi)

            batch = pipeline.request_batch(request)

            rasterized = batch.arrays[ArrayKeys.RASTERIZED].data

            # check larger radius: rasterized point (0, 0, 0) should extend in
            # x,y by 10; z, by 1
            self.assertEqual(rasterized[0, 10, 0], 1)
            self.assertEqual(rasterized[0, 11, 0], 0)
            self.assertEqual(rasterized[0, 0, 5], 1)
            self.assertEqual(rasterized[0, 0, 6], 0)
            self.assertEqual(rasterized[1, 0, 0], 1)
            self.assertEqual(rasterized[2, 0, 0], 0)

        # same with anisotropic radius and inner radius

        pipeline = (
            PointTestSource3D() +
            RasterizePoints(
                PointsKeys.TEST_POINTS,
                ArrayKeys.RASTERIZED,
                ArraySpec(voxel_size=(40, 4, 4)),
                RasterizationSettings(
                    radius=(40, 40, 20),
                    inner_radius_fraction=0.75,
                    fg_value=1,
                    bg_value=0))
        )

        with build(pipeline):
            request = BatchRequest()
            roi = Roi((0, 0, 0), (120, 80, 80))

            request[PointsKeys.TEST_POINTS] = PointsSpec(roi=roi)
            request[ArrayKeys.GT_LABELS] = ArraySpec(roi=roi)
            request[ArrayKeys.RASTERIZED] = ArraySpec(roi=roi)

            batch = pipeline.request_batch(request)

            rasterized = batch.arrays[ArrayKeys.RASTERIZED].data

            # in the middle of the ball, there should be 0 (since inner radius is set)
            self.assertEqual(rasterized[0, 0, 0], 0)
            # check larger radius: rasterized point (0, 0, 0) should extend in
            # x,y by 10; z, by 1
            self.assertEqual(rasterized[0, 10, 0], 1)
            self.assertEqual(rasterized[0, 11, 0], 0)
            self.assertEqual(rasterized[0, 0, 5], 1)
            self.assertEqual(rasterized[0, 0, 6], 0)
            self.assertEqual(rasterized[1, 0, 0], 1)
            self.assertEqual(rasterized[2, 0, 0], 0)
