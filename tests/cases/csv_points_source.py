import csv
import random

import numpy as np
import pytest

from gunpowder import (
    BatchRequest,
    Coordinate,
    CsvPointsSource,
    GraphKey,
    GraphSpec,
    Roi,
    build,
)


# automatically set the seed for all tests
@pytest.fixture(autouse=True)
def seeds():
    random.seed(12345)
    np.random.seed(12345)


@pytest.fixture
def test_points_2d(tmpdir):
    fake_points_file = tmpdir / "shift_test.csv"
    fake_points = np.random.randint(0, 100, size=(2, 2))
    with open(fake_points_file, "w") as f:
        for point in fake_points:
            f.write(str(point[0]) + "\t" + str(point[1]) + "\n")

    yield fake_points_file, fake_points


@pytest.fixture
def test_points_3d(tmpdir):
    fake_points_file = tmpdir / "shift_test.csv"
    fake_points = np.random.randint(0, 100, size=(3, 3)).astype(float)
    with open(fake_points_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["x", "y", "z", "id"])
        writer.writeheader()
        for i, point in enumerate(fake_points):
            pointdict = {"x": point[0], "y": point[1], "z": point[2], "id": i}
            writer.writerow(pointdict)

    yield fake_points_file, fake_points


def test_pipeline_2d(test_points_2d):
    fake_points_file, fake_points = test_points_2d

    points_key = GraphKey("TEST_POINTS")

    csv_source = CsvPointsSource(
        fake_points_file,
        points_key,
        spatial_cols=[0, 1],
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

    assert sorted(result_locs) == sorted(target_locs)


def test_pipeline_3d(test_points_3d):
    fake_points_file, fake_points = test_points_3d

    points_key = GraphKey("TEST_POINTS")
    scale = 2
    csv_source = CsvPointsSource(
        fake_points_file,
        points_key,
        spatial_cols=[0, 2, 1],
        delimiter=",",
        id_col=3,
        points_spec=GraphSpec(roi=Roi(shape=Coordinate((100, 100)), offset=(0, 0))),
        scale=scale,
    )

    request = BatchRequest()
    shape = Coordinate((100, 100, 100))
    request.add(points_key, shape)

    pipeline = csv_source
    with build(pipeline) as b:
        request = b.request_batch(request)

    result_points = list(request[points_key].nodes)
    for node in result_points:
        orig_loc = fake_points[int(node.id)]
        reordered_loc = orig_loc.copy()
        reordered_loc[1] = orig_loc[2]
        reordered_loc[2] = orig_loc[1]
        assert list(node.location) == list(reordered_loc * scale)
