from gunpowder import (
    BatchProvider,
    BatchRequest,
    Batch,
    Roi,
    Coordinate,
    PointsSpec,
    PointsKey,
    ArrayKeys,
    ArrayKey,
    ArraySpec,
    Array,
    ElasticAugment,
    RandomLocation,
    Snapshot,
    build,
)
from gunpowder.points import Points, PointsKeys, Point
from .provider_test import ProviderTest

import pytest
import numpy as np

import math
import copy


class PointTestSource3D(BatchProvider):
    def setup(self):

        self.points = {
            0: Point([0, 10, 0]),
            1: Point([0, 30, 0]),
            2: Point([0, 50, 0]),
            3: Point([0, 70, 0]),
            4: Point([0, 90, 0]),
        }

        self.provides(
            PointsKeys.TEST_POINTS,
            PointsSpec(roi=Roi((-100, -100, -100), (300, 300, 300))),
        )

        self.provides(
            ArrayKeys.TEST_LABELS,
            ArraySpec(
                roi=Roi((-100, -100, -100), (300, 300, 300)),
                voxel_size=Coordinate((4, 1, 1)),
                interpolatable=False,
            ),
        )

    def point_to_voxel(self, array_roi, location):

        # location is in world units, get it into voxels
        location = location / self.spec[ArrayKeys.TEST_LABELS].voxel_size

        # shift location relative to beginning of array roi
        location -= array_roi.get_begin() / self.spec[ArrayKeys.TEST_LABELS].voxel_size

        return tuple(slice(int(l - 2), int(l + 3)) for l in location)

    def provide(self, request):

        batch = Batch()

        if PointsKeys.TEST_POINTS in request:
            roi_points = request[PointsKeys.TEST_POINTS].roi

            points = {}
            for i, point in self.points.items():
                if roi_points.contains(point.location):
                    points[i] = copy.deepcopy(point)
            batch.points[PointsKeys.TEST_POINTS] = Points(
                points, PointsSpec(roi=roi_points)
            )

        if ArrayKeys.TEST_LABELS in request:
            roi_array = request[ArrayKeys.TEST_LABELS].roi
            roi_voxel = roi_array // self.spec[ArrayKeys.TEST_LABELS].voxel_size

            data = np.zeros(roi_voxel.get_shape(), dtype=np.uint32)
            data[:, ::2] = 100

            for i, point in self.points.items():
                loc = self.point_to_voxel(roi_array, point.location)
                data[loc] = i

            spec = self.spec[ArrayKeys.TEST_LABELS].copy()
            spec.roi = roi_array
            batch.arrays[ArrayKeys.TEST_LABELS] = Array(data, spec=spec)

        return batch


class TestPlaceholderRequest(ProviderTest):
    def test_without_placeholder(self):

        test_labels = ArrayKey("TEST_LABELS")
        test_points = PointsKey("TEST_POINTS")

        pipeline = (
            PointTestSource3D()
            + RandomLocation(ensure_nonempty=test_points)
            + ElasticAugment([10, 10, 10], [0.1, 0.1, 0.1], [0, 2.0 * math.pi])
            + Snapshot(
                {test_labels: "volumes/labels"},
                output_dir=self.path_to(),
                output_filename="elastic_augment_test{id}-{iteration}.hdf",
            )
        )

        with build(pipeline):
            for i in range(100):

                request_size = Coordinate((40, 40, 40))

                request_a = BatchRequest(random_seed=i)
                request_a.add(test_points, request_size)

                request_b = BatchRequest(random_seed=i)
                request_b.add(test_points, request_size)
                request_b.add(test_labels, request_size)

                # No array to provide a voxel size to ElasticAugment
                with pytest.raises(RuntimeError):
                    pipeline.request_batch(request_a)
                batch_b = pipeline.request_batch(request_b)

                self.assertIn(test_labels, batch_b)

    def test_placeholder(self):

        test_labels = ArrayKey("TEST_LABELS")
        test_points = PointsKey("TEST_POINTS")

        pipeline = (
            PointTestSource3D()
            + RandomLocation(ensure_nonempty=test_points)
            + ElasticAugment([10, 10, 10], [0.1, 0.1, 0.1], [0, 2.0 * math.pi])
            + Snapshot(
                {test_labels: "volumes/labels"},
                output_dir=self.path_to(),
                output_filename="elastic_augment_test{id}-{iteration}.hdf",
            )
        )

        with build(pipeline):
            for i in range(100):

                request_size = Coordinate((40, 40, 40))

                request_a = BatchRequest(random_seed=i)
                request_a.add(test_points, request_size)
                request_a.add(test_labels, request_size, placeholder=True)

                request_b = BatchRequest(random_seed=i)
                request_b.add(test_points, request_size)
                request_b.add(test_labels, request_size)

                batch_a = pipeline.request_batch(request_a)
                batch_b = pipeline.request_batch(request_b)

                points_a = batch_a[test_points].nodes
                points_b = batch_b[test_points].nodes

                for a, b in zip(points_a, points_b):
                    assert all(np.isclose(a.location, b.location))