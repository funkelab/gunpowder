import numpy as np
import pytest

from gunpowder import (
    Batch,
    BatchProvider,
    BatchRequest,
    Coordinate,
    Graph,
    GraphKey,
    GraphSpec,
    Node,
    RandomLocation,
    Roi,
    build,
)


class ExampleSourceRandomLocation(BatchProvider):
    def __init__(self, points_key):
        self.points_key = points_key
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
        self.provides(self.points_key, self.graph.spec)

    def provide(self, request):
        batch = Batch()

        roi = request[self.points_key].roi
        batch[self.points_key] = self.graph.crop(roi).trim(roi)
        return batch


def test_output():
    points_key = GraphKey("TEST_POINTS")

    pipeline = ExampleSourceRandomLocation(points_key) + RandomLocation(
        ensure_nonempty=points_key, point_balance_radius=100
    )

    # count the number of times we get each point
    histogram = {}

    with build(pipeline):
        for i in range(500):
            batch = pipeline.request_batch(
                BatchRequest(
                    {points_key: GraphSpec(roi=Roi((0, 0, 0), (100, 100, 100)))}
                )
            )

            points = {node.id: node for node in batch[points_key].nodes}

            assert len(points) > 0
            assert (1 in points) != (2 in points or 3 in points)

            for node in batch[points_key].nodes:
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
            assert abs(histogram[i] - histogram[j]) < 1


def test_equal_probability():
    points_key = GraphKey("TEST_POINTS")

    pipeline = ExampleSourceRandomLocation(points_key) + RandomLocation(
        ensure_nonempty=points_key
    )

    # count the number of times we get each point
    histogram = {}

    with build(pipeline):
        for i in range(500):
            batch = pipeline.request_batch(
                BatchRequest({points_key: GraphSpec(roi=Roi((0, 0, 0), (10, 10, 10)))})
            )

            points = {node.id: node for node in batch[points_key].nodes}

            assert len(points) > 0
            assert (1 in points) != (2 in points or 3 in points)

            for point in batch[points_key].nodes:
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
            assert abs(histogram[i] - histogram[j]) < 1


@pytest.mark.xfail
def test_ensure_centered():
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

    points_key = GraphKey("TEST_POINTS")

    pipeline = ExampleSourceRandomLocation(points_key) + RandomLocation(
        ensure_nonempty=points_key, ensure_centered=True
    )

    # count the number of times we get each point
    histogram = {}

    with build(pipeline):
        for i in range(500):
            batch = pipeline.request_batch(
                BatchRequest(
                    {points_key: GraphSpec(roi=Roi((0, 0, 0), (100, 100, 100)))}
                )
            )

            points = {node.id: node for node in batch[points_key].nodes}
            roi = batch[points_key].spec.roi

            locations = tuple([Coordinate(point.location) for point in points.values()])
            assert Coordinate([50, 50, 50]) in locations

            assert len(points) > 0
            assert (1 in points) != (2 in points or 3 in points)

            for point_id in batch[points_key].data.keys():
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
            assert abs(histogram[i] - histogram[j]) < 1
