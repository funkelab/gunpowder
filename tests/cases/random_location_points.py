from .provider_test import ProviderTest
from gunpowder import (
    BatchProvider,
    BatchRequest,
    Batch,
    Node,
    Graph,
    GraphSpec,
    GraphKey,
    GraphKeys,
    RandomLocation,
    build,
    Roi,
    Coordinate,
)

import numpy as np
import pytest

import unittest


class ExampleSourceRandomLocation(BatchProvider):
    def __init__(self):
        self.graph = Graph(
            [
                Node(1, np.array([1, 1, 1])),
                Node(2, np.array([500, 500, 500])),
                Node(3, np.array([550, 550, 550])),
            ],
            [],
            GraphSpec(roi=Roi((-500, -500, -500), (1500, 1500, 1500))),
        )

    def setup(self):
        self.provides(GraphKeys.TEST_POINTS, self.graph.spec)

    def provide(self, request):
        batch = Batch()

        roi = request[GraphKeys.TEST_POINTS].roi
        batch[GraphKeys.TEST_POINTS] = self.graph.crop(roi).trim(roi)
        return batch


class TestRandomLocationPoints(ProviderTest):
    @pytest.mark.xfail
    def test_output(self):
        """
        Fails due to probabilities being calculated in advance, rather than after creating
        each roi. The new approach does not account for all possible roi's containing
        each point, some of which may not contain its nearest neighbors.
        """

        GraphKey("TEST_POINTS")

        pipeline = ExampleSourceRandomLocation() + RandomLocation(
            ensure_nonempty=GraphKeys.TEST_POINTS, point_balance_radius=100
        )

        # count the number of times we get each point
        histogram = {}

        with build(pipeline):
            for i in range(5000):
                batch = pipeline.request_batch(
                    BatchRequest(
                        {
                            GraphKeys.TEST_POINTS: GraphSpec(
                                roi=Roi((0, 0, 0), (100, 100, 100))
                            )
                        }
                    )
                )

                points = {node.id: node for node in batch[GraphKeys.TEST_POINTS].nodes}

                self.assertTrue(len(points) > 0)
                self.assertTrue((1 in points) != (2 in points or 3 in points), points)

                for node in batch[GraphKeys.TEST_POINTS].nodes:
                    if node.id not in histogram:
                        histogram[node.id] = 1
                    else:
                        histogram[node.id] += 1

        total = sum(histogram.values())
        for k, v in histogram.items():
            histogram[k] = float(v) / total

        # we should get roughly the same count for each point
        for i in histogram.keys():
            for j in histogram.keys():
                self.assertAlmostEqual(histogram[i], histogram[j], 1)

    def test_equal_probability(self):
        GraphKey("TEST_POINTS")

        pipeline = ExampleSourceRandomLocation() + RandomLocation(
            ensure_nonempty=GraphKeys.TEST_POINTS
        )

        # count the number of times we get each point
        histogram = {}

        with build(pipeline):
            for i in range(5000):
                batch = pipeline.request_batch(
                    BatchRequest(
                        {
                            GraphKeys.TEST_POINTS: GraphSpec(
                                roi=Roi((0, 0, 0), (10, 10, 10))
                            )
                        }
                    )
                )

                points = {node.id: node for node in batch[GraphKeys.TEST_POINTS].nodes}

                self.assertTrue(len(points) > 0)
                self.assertTrue((1 in points) != (2 in points or 3 in points), points)

                for point in batch[GraphKeys.TEST_POINTS].nodes:
                    if point.id not in histogram:
                        histogram[point.id] = 1
                    else:
                        histogram[point.id] += 1

        total = sum(histogram.values())
        for k, v in histogram.items():
            histogram[k] = float(v) / total

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

        GraphKey("TEST_POINTS")

        pipeline = ExampleSourceRandomLocation() + RandomLocation(
            ensure_nonempty=GraphKeys.TEST_POINTS, ensure_centered=True
        )

        # count the number of times we get each point
        histogram = {}

        with build(pipeline):
            for i in range(5000):
                batch = pipeline.request_batch(
                    BatchRequest(
                        {
                            GraphKeys.TEST_POINTS: GraphSpec(
                                roi=Roi((0, 0, 0), (100, 100, 100))
                            )
                        }
                    )
                )

                points = batch[GraphKeys.TEST_POINTS].data
                roi = batch[GraphKeys.TEST_POINTS].spec.roi

                locations = tuple(
                    [Coordinate(point.location) for point in points.values()]
                )
                self.assertTrue(
                    Coordinate([50, 50, 50]) in locations,
                    f"locations: {tuple([point.location for point in points.values()])}",
                )

                self.assertTrue(len(points) > 0)
                self.assertTrue((1 in points) != (2 in points or 3 in points), points)

                for point_id in batch[GraphKeys.TEST_POINTS].data.keys():
                    if point_id not in histogram:
                        histogram[point_id] = 1
                    else:
                        histogram[node.id] += 1

        total = sum(histogram.values())
        for k, v in histogram.items():
            histogram[k] = float(v) / total

        # we should get roughly the same count for each point
        for i in histogram.keys():
            for j in histogram.keys():
                self.assertAlmostEqual(histogram[i], histogram[j], 1, histogram)
