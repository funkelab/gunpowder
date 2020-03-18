from .provider_test import ProviderTest
import numpy as np
import unittest
from gunpowder import (
    BatchProvider,
    BatchRequest,
    Batch,
    Point,
    Points,
    PointsSpec,
    PointsKey,
    PointsKeys,
    Node,
    RandomLocation,
    build,
    Roi,
    Coordinate,
)


class TestSourceRandomLocation(BatchProvider):

    def __init__(self):

        self.points = Points(
            {
                1: Point(np.array([1, 1, 1])),
                2: Point(np.array([500, 500, 500])),
                3: Point(np.array([550, 550, 550])),
            },
            PointsSpec(
                roi=Roi((-500, -500, -500), (1500, 1500, 1500))))

    def setup(self):

        self.provides(
            PointsKeys.TEST_POINTS,
            self.points.spec)

    def provide(self, request):

        batch = Batch()

        roi = request[PointsKeys.TEST_POINTS].roi
        batch[PointsKeys.TEST_POINTS] = self.points.crop(roi).trim(roi)
        return batch


class TestRandomLocationPoints(ProviderTest):

    def test_output(self):

        PointsKey('TEST_POINTS')

        pipeline = (
            TestSourceRandomLocation() +
            RandomLocation(ensure_nonempty=PointsKeys.TEST_POINTS)
        )

        # count the number of times we get each point
        histogram = {}

        with build(pipeline):

            for i in range(5000):
                batch = pipeline.request_batch(
                    BatchRequest(
                        {
                            PointsKeys.TEST_POINTS: PointsSpec(
                                roi=Roi((0, 0, 0), (100, 100, 100)))
                        }))

                points = batch[PointsKeys.TEST_POINTS].data

                self.assertTrue(len(points) > 0)
                self.assertTrue((1 in points) != (2 in points or 3 in points), points)

                for point_id in batch[PointsKeys.TEST_POINTS].data.keys():
                    if point_id not in histogram:
                        histogram[point_id] = 1
                    else:
                        histogram[point_id] += 1

        total = sum(histogram.values())
        for k, v in histogram.items():
            histogram[k] = float(v)/total

        # we should get roughly the same count for each point
        for i in histogram.keys():
            for j in histogram.keys():
                self.assertAlmostEqual(histogram[i], histogram[j], 1)

    @unittest.expectedFailure
    def test_ensure_centered(self):
        """
        Expected failure due to emergent behavior of two desired rules:
        1) Points on the upper bound of Roi are not considered contained
        2) When considering a point as a center of a random location,
            scale by the number of points within some delta distance

        if two points are equally likely to be chosen, and centering
        a roi on either of them means the other is on the bounding box
        of the roi, then it can be the case that if the roi is centered
        one of them, the roi contains only that one, but if the roi is
        centered on the second, then both are considered contained,
        breaking the equal likelihood of picking each point.
        """

        PointsKey("TEST_POINTS")

        pipeline = TestSourceRandomLocation() + RandomLocation(
            ensure_nonempty=PointsKeys.TEST_POINTS, ensure_centered=True
        )

        # count the number of times we get each point
        histogram = {}

        with build(pipeline):

            for i in range(5000):
                batch = pipeline.request_batch(
                    BatchRequest(
                        {
                            PointsKeys.TEST_POINTS: PointsSpec(
                                roi=Roi((0, 0, 0), (100, 100, 100))
                            )
                        }
                    )
                )

                points = batch[PointsKeys.TEST_POINTS].data
                roi = batch[PointsKeys.TEST_POINTS].spec.roi

                locations = tuple([Coordinate(point.location) for point in points.values()])
                self.assertTrue(
                    Coordinate([50, 50, 50]) in locations,
                    f"locations: {tuple([point.location for point in points.values()])}"
                )

                self.assertTrue(len(points) > 0)
                self.assertTrue((1 in points) != (2 in points or 3 in points), points)

                for point_id in batch[PointsKeys.TEST_POINTS].data.keys():
                    if point_id not in histogram:
                        histogram[point_id] = 1
                    else:
                        histogram[point_id] += 1

        total = sum(histogram.values())
        for k, v in histogram.items():
            histogram[k] = float(v) / total

        # we should get roughly the same count for each point
        for i in histogram.keys():
            for j in histogram.keys():
                self.assertAlmostEqual(histogram[i], histogram[j], 1, histogram)
