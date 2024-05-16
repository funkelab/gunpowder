import random

import numpy as np
import pytest
from pytest_unordered import unordered
import unittest

from gunpowder import (
    ArraySpec,
    BatchRequest,
    CsvPointsSource,
    GraphKey,
    GraphSpec,
    build,
    Coordinate,
    Roi,
)


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
    fake_points = np.random.randint(0, 100, size=(2, 2))
    with open(fake_points_file, "w") as f:
        for point in fake_points:
            f.write(str(point[0]) + "\t" + str(point[1]) + "\n")

    # This fixture will run after seeds since it is set
    # with autouse=True. So make sure to reset the seeds properly at the end
    # of this fixture
    random.seed(12345)
    np.random.seed(12345)

    yield fake_points_file, fake_points


def test_pipeline3(test_points):
    fake_points_file, fake_points = test_points

    points_key = GraphKey("TEST_POINTS")
    voxel_size = Coordinate((1, 1))
    spec = ArraySpec(voxel_size=voxel_size, interpolatable=True)

    csv_source = CsvPointsSource(
        fake_points_file,
        points_key,
        spatial_cols=[
            0,
            1,
        ],
        delimiter="\t",
        points_spec=GraphSpec(roi=Roi(shape=Coordinate((100, 100)), offset=(0, 0))),
    )

    request = BatchRequest()
    shape = Coordinate((100, 100))
    request.add(points_key, shape)

    pipeline = csv_source
    with build(pipeline) as b:
        request = b.request_batch(request)

    target_locs = [list(fake_point) for fake_point in fake_points]
    result_points = list(request[points_key].nodes)
    result_locs = [list(point.location) for point in result_points]

    assert result_locs == unordered(target_locs)
