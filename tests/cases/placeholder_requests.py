from gunpowder import (
    PipelineRequestError,
    BatchProvider,
    BatchRequest,
    Batch,
    Roi,
    Coordinate,
    GraphSpec,
    GraphKey,
    ArrayKeys,
    ArrayKey,
    ArraySpec,
    Array,
    ElasticAugment,
    RandomLocation,
    Snapshot,
    build,
)
from gunpowder.graph import Graph, GraphKeys, Node
from .provider_test import ProviderTest

import pytest
import numpy as np

import math
import copy


class PointTestSource3D(BatchProvider):
    def setup(self):
        self.points = [
            Node(0, np.array([0, 10, 0])),
            Node(1, np.array([0, 30, 0])),
            Node(2, np.array([0, 50, 0])),
            Node(3, np.array([0, 70, 0])),
            Node(4, np.array([0, 90, 0])),
        ]

        self.provides(
            GraphKeys.TEST_POINTS,
            GraphSpec(roi=Roi((-100, -100, -100), (300, 300, 300))),
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
        location -= array_roi.begin / self.spec[ArrayKeys.TEST_LABELS].voxel_size

        return tuple(slice(int(l - 2), int(l + 3)) for l in location)

    def provide(self, request):
        batch = Batch()

        if GraphKeys.TEST_POINTS in request:
            roi_points = request[GraphKeys.TEST_POINTS].roi

            contained_points = []
            for point in self.points:
                if roi_points.contains(point.location):
                    contained_points.append(copy.deepcopy(point))
            batch[GraphKeys.TEST_POINTS] = Graph(
                contained_points, [], GraphSpec(roi=roi_points)
            )

        if ArrayKeys.TEST_LABELS in request:
            roi_array = request[ArrayKeys.TEST_LABELS].roi
            roi_voxel = roi_array // self.spec[ArrayKeys.TEST_LABELS].voxel_size

            data = np.zeros(roi_voxel.shape, dtype=np.uint32)
            data[:, ::2] = 100

            for point in self.points:
                loc = self.point_to_voxel(roi_array, point.location)
                data[loc] = point.id

            spec = self.spec[ArrayKeys.TEST_LABELS].copy()
            spec.roi = roi_array
            batch.arrays[ArrayKeys.TEST_LABELS] = Array(data, spec=spec)

        return batch


class TestPlaceholderRequest(ProviderTest):
    def test_without_placeholder(self):
        test_labels = ArrayKey("TEST_LABELS")
        test_points = GraphKey("TEST_POINTS")

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
            for i in range(2):
                request_size = Coordinate((40, 40, 40))

                request_a = BatchRequest(random_seed=i)
                request_a.add(test_points, request_size)

                request_b = BatchRequest(random_seed=i)
                request_b.add(test_points, request_size)
                request_b.add(test_labels, request_size)

                # No array to provide a voxel size to ElasticAugment
                with pytest.raises(PipelineRequestError):
                    pipeline.request_batch(request_a)
                batch_b = pipeline.request_batch(request_b)

                self.assertIn(test_labels, batch_b)

    def test_placeholder(self):
        test_labels = ArrayKey("TEST_LABELS")
        test_points = GraphKey("TEST_POINTS")

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
            for i in range(2):
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
