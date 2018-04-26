from .provider_test import ProviderTest
from gunpowder import *
import numpy as np

class TestSourceRandomLocation(BatchProvider):

    def __init__(self):

        self.points = Points(
            {
                1: Point([1, 1, 1]),
                2: Point([500, 500, 500]),
                3: Point([550, 550, 550]),
            },
            PointsSpec(
                roi=Roi((0, 0, 0), (1000, 1000, 1000))))

    def setup(self):

        self.provides(
            PointsKeys.TEST_POINTS,
            self.points.spec)

    def provide(self, request):

        batch = Batch()

        roi = request[PointsKeys.TEST_POINTS].roi
        points = Points({}, PointsSpec(roi))

        for point_id, point in self.points.data.items():
            if roi.contains(point.location):
                points.data[point_id] = point.copy()
        batch[PointsKeys.TEST_POINTS] = points

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
