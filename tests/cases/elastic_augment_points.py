import unittest
from gunpowder import (
    BatchProvider,
    Batch,
    BatchRequest,
    GraphSpec,
    GraphKeys,
    GraphKey,
    Graph,
    Node,
    ArraySpec,
    ArrayKeys,
    ArrayKey,
    Array,
    Roi,
    Coordinate,
    ElasticAugment,
    RasterizeGraph,
    RasterizationSettings,
    Snapshot,
    build,
)
from .provider_test import ProviderTest

import numpy as np
import math
import time
import unittest


class PointTestSource3D(BatchProvider):
    def setup(self):
        self.points = [
            Node(0, np.array([0, 0, 0])),
            Node(1, np.array([0, 10, 0])),
            Node(2, np.array([0, 20, 0])),
            Node(3, np.array([0, 30, 0])),
            Node(4, np.array([0, 40, 0])),
            Node(5, np.array([0, 50, 0])),
        ]

        self.provides(
            GraphKeys.TEST_POINTS,
            GraphSpec(roi=Roi((-100, -100, -100), (200, 200, 200))),
        )

        self.provides(
            ArrayKeys.TEST_LABELS,
            ArraySpec(
                roi=Roi((-100, -100, -100), (200, 200, 200)),
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

        roi_points = request[GraphKeys.TEST_POINTS].roi
        roi_array = request[ArrayKeys.TEST_LABELS].roi
        roi_voxel = roi_array // self.spec[ArrayKeys.TEST_LABELS].voxel_size

        data = np.zeros(roi_voxel.shape, dtype=np.uint32)
        data[:, ::2] = 100

        for node in self.points:
            loc = self.point_to_voxel(roi_array, node.location)
            data[loc] = node.id

        spec = self.spec[ArrayKeys.TEST_LABELS].copy()
        spec.roi = roi_array
        batch.arrays[ArrayKeys.TEST_LABELS] = Array(data, spec=spec)

        points = []
        for node in self.points:
            if roi_points.contains(node.location):
                points.append(node)
        batch.graphs[GraphKeys.TEST_POINTS] = Graph(
            points, [], GraphSpec(roi=roi_points)
        )

        return batch


class DensePointTestSource3D(BatchProvider):
    def setup(self):
        self.points = [
            Node(i, np.array([(i // 100) % 10 * 4, (i // 10) % 10 * 4, i % 10 * 4]))
            for i in range(1000)
        ]

        self.provides(
            GraphKeys.TEST_POINTS,
            GraphSpec(roi=Roi((-40, -40, -40), (120, 120, 120))),
        )

        self.provides(
            ArrayKeys.TEST_LABELS,
            ArraySpec(
                roi=Roi((-40, -40, -40), (120, 120, 120)),
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

        roi_points = request[GraphKeys.TEST_POINTS].roi
        roi_array = request[ArrayKeys.TEST_LABELS].roi
        roi_voxel = roi_array // self.spec[ArrayKeys.TEST_LABELS].voxel_size

        data = np.zeros(roi_voxel.shape, dtype=np.uint32)
        data[:, ::2] = 100

        for node in self.points:
            loc = self.point_to_voxel(roi_array, node.location)
            data[loc] = node.id

        spec = self.spec[ArrayKeys.TEST_LABELS].copy()
        spec.roi = roi_array
        batch.arrays[ArrayKeys.TEST_LABELS] = Array(data, spec=spec)

        points = []
        for point in self.points:
            if roi_points.contains(point.location):
                points.append(point)
        batch[GraphKeys.TEST_POINTS] = Graph(points, [], GraphSpec(roi=roi_points))

        return batch


class TestElasticAugment(ProviderTest):
    def test_3d_basics(self):
        test_labels = ArrayKey("TEST_LABELS")
        test_points = GraphKey("TEST_POINTS")
        test_raster = ArrayKey("TEST_RASTER")

        pipeline = (
            PointTestSource3D()
            + ElasticAugment(
                [10, 10, 10],
                [0.1, 0.1, 0.1],
                # [0, 0, 0], # no jitter
                [0, 2.0 * math.pi],
            )
            + RasterizeGraph(
                test_points,
                test_raster,
                settings=RasterizationSettings(radius=2, mode="peak"),
            )
            + Snapshot(
                {test_labels: "volumes/labels", test_raster: "volumes/raster"},
                dataset_dtypes={test_raster: np.float32},
                output_dir=self.path_to(),
                output_filename="elastic_augment_test{id}-{iteration}.hdf",
            )
        )

        for _ in range(5):
            with build(pipeline):
                request_roi = Roi((-20, -20, -20), (40, 40, 40))

                request = BatchRequest()
                request[test_labels] = ArraySpec(roi=request_roi)
                request[test_points] = GraphSpec(roi=request_roi)
                request[test_raster] = ArraySpec(roi=request_roi)

                batch = pipeline.request_batch(request)
                labels = batch[test_labels]
                points = batch[test_points]

                # the point at (0, 0, 0) should not have moved
                self.assertTrue(points.contains(0))

                labels_data_roi = (
                    labels.spec.roi - labels.spec.roi.begin
                ) / labels.spec.voxel_size

                # points should have moved together with the voxels
                for point in points.nodes:
                    loc = point.location - labels.spec.roi.begin
                    loc = loc / labels.spec.voxel_size
                    loc = Coordinate(int(round(x)) for x in loc)
                    if labels_data_roi.contains(loc):
                        self.assertEqual(labels.data[loc], point.id)

    def test_random_seed(self):
        test_labels = ArrayKey("TEST_LABELS")
        test_points = GraphKey("TEST_POINTS")
        test_raster = ArrayKey("TEST_RASTER")

        pipeline = (
            PointTestSource3D()
            + ElasticAugment(
                [10, 10, 10],
                [0.1, 0.1, 0.1],
                # [0, 0, 0], # no jitter
                [0, 2.0 * math.pi],
            )
            +  # rotate randomly
            # [math.pi/4, math.pi/4]) + # rotate by 45 deg
            # [0, 0]) + # no rotation
            RasterizeGraph(
                test_points,
                test_raster,
                settings=RasterizationSettings(radius=2, mode="peak"),
            )
            + Snapshot(
                {test_labels: "volumes/labels", test_raster: "volumes/raster"},
                dataset_dtypes={test_raster: np.float32},
                output_dir=self.path_to(),
                output_filename="elastic_augment_test{id}-{iteration}.hdf",
            )
        )

        batch_points = []
        for _ in range(5):
            with build(pipeline):
                request_roi = Roi((-20, -20, -20), (40, 40, 40))

                request = BatchRequest(random_seed=10)
                request[test_labels] = ArraySpec(roi=request_roi)
                request[test_points] = GraphSpec(roi=request_roi)
                request[test_raster] = ArraySpec(roi=request_roi)
                batch = pipeline.request_batch(request)
                labels = batch[test_labels]
                points = batch[test_points]
                batch_points.append(
                    tuple((node.id, tuple(node.location)) for node in points.nodes)
                )

                # the point at (0, 0, 0) should not have moved
                data = {node.id: node for node in points.nodes}
                self.assertTrue(0 in data)

                labels_data_roi = (
                    labels.spec.roi - labels.spec.roi.begin
                ) / labels.spec.voxel_size

                # points should have moved together with the voxels
                for node in points.nodes:
                    loc = node.location - labels.spec.roi.begin
                    loc = loc / labels.spec.voxel_size
                    loc = Coordinate(int(round(x)) for x in loc)
                    if labels_data_roi.contains(loc):
                        self.assertEqual(labels.data[loc], node.id)

        for point_data in zip(*batch_points):
            self.assertEqual(len(set(point_data)), 1)

    def test_fast_transform(self):
        test_labels = ArrayKey("TEST_LABELS")
        test_points = GraphKey("TEST_POINTS")
        test_raster = ArrayKey("TEST_RASTER")
        fast_pipeline = (
            DensePointTestSource3D()
            + ElasticAugment(
                [10, 10, 10],
                [0.1, 0.1, 0.1],
                [0, 2.0 * math.pi],
                use_fast_points_transform=True,
            )
            + RasterizeGraph(
                test_points,
                test_raster,
                settings=RasterizationSettings(radius=2, mode="peak"),
            )
        )

        reference_pipeline = (
            DensePointTestSource3D()
            + ElasticAugment([10, 10, 10], [0.1, 0.1, 0.1], [0, 2.0 * math.pi])
            + RasterizeGraph(
                test_points,
                test_raster,
                settings=RasterizationSettings(radius=2, mode="peak"),
            )
        )

        timings = []
        for i in range(5):
            points_fast = {}
            points_reference = {}
            # seed chosen specifically to make this test fail
            seed = i + 15
            with build(fast_pipeline):
                request_roi = Roi((0, 0, 0), (40, 40, 40))

                request = BatchRequest(random_seed=seed)
                request[test_labels] = ArraySpec(roi=request_roi)
                request[test_points] = GraphSpec(roi=request_roi)
                request[test_raster] = ArraySpec(roi=request_roi)

                t1_fast = time.time()
                batch = fast_pipeline.request_batch(request)
                t2_fast = time.time()
                points_fast = {node.id: node for node in batch[test_points].nodes}

            with build(reference_pipeline):
                request_roi = Roi((0, 0, 0), (40, 40, 40))

                request = BatchRequest(random_seed=seed)
                request[test_labels] = ArraySpec(roi=request_roi)
                request[test_points] = GraphSpec(roi=request_roi)
                request[test_raster] = ArraySpec(roi=request_roi)

                t1_ref = time.time()
                batch = reference_pipeline.request_batch(request)
                t2_ref = time.time()
                points_reference = {node.id: node for node in batch[test_points].nodes}

            timings.append((t2_fast - t1_fast, t2_ref - t1_ref))
            diffs = []
            missing = 0
            for point_id, point in points_reference.items():
                if point_id not in points_fast:
                    missing += 1
                    continue
                diff = point.location - points_fast[point_id].location
                diffs.append(tuple(diff))
                self.assertAlmostEqual(
                    np.linalg.norm(diff),
                    0,
                    delta=1,
                    msg="fast transform returned location {} but expected {} for point {}".format(
                        point.location, points_fast[point_id].location, point_id
                    ),
                )

            t_fast, t_ref = [np.mean(x) for x in zip(*timings)]
            self.assertLess(t_fast, t_ref)
            self.assertEqual(missing, 0)

    def test_fast_transform_no_recompute(self):
        test_labels = ArrayKey("TEST_LABELS")
        test_points = GraphKey("TEST_POINTS")
        test_raster = ArrayKey("TEST_RASTER")
        fast_pipeline = (
            DensePointTestSource3D()
            + ElasticAugment(
                [10, 10, 10],
                [0.1, 0.1, 0.1],
                [0, 2.0 * math.pi],
                use_fast_points_transform=True,
                recompute_missing_points=False,
            )
            + RasterizeGraph(
                test_points,
                test_raster,
                settings=RasterizationSettings(radius=2, mode="peak"),
            )
        )

        reference_pipeline = (
            DensePointTestSource3D()
            + ElasticAugment([10, 10, 10], [0.1, 0.1, 0.1], [0, 2.0 * math.pi])
            + RasterizeGraph(
                test_points,
                test_raster,
                settings=RasterizationSettings(radius=2, mode="peak"),
            )
        )

        timings = []
        for i in range(5):
            points_fast = {}
            points_reference = {}
            # seed chosen specifically to make this test fail
            seed = i + 15
            with build(fast_pipeline):
                request_roi = Roi((0, 0, 0), (40, 40, 40))

                request = BatchRequest(random_seed=seed)
                request[test_labels] = ArraySpec(roi=request_roi)
                request[test_points] = GraphSpec(roi=request_roi)
                request[test_raster] = ArraySpec(roi=request_roi)

                t1_fast = time.time()
                batch = fast_pipeline.request_batch(request)
                t2_fast = time.time()
                points_fast = {node.id: node for node in batch[test_points].nodes}

            with build(reference_pipeline):
                request_roi = Roi((0, 0, 0), (40, 40, 40))

                request = BatchRequest(random_seed=seed)
                request[test_labels] = ArraySpec(roi=request_roi)
                request[test_points] = GraphSpec(roi=request_roi)
                request[test_raster] = ArraySpec(roi=request_roi)

                t1_ref = time.time()
                batch = reference_pipeline.request_batch(request)
                t2_ref = time.time()
                points_reference = {node.id: node for node in batch[test_points].nodes}

            timings.append((t2_fast - t1_fast, t2_ref - t1_ref))
            diffs = []
            missing = 0
            for point_id, point in points_reference.items():
                if point_id not in points_fast:
                    missing += 1
                    continue
                diff = point.location - points_fast[point_id].location
                diffs.append(tuple(diff))
                self.assertAlmostEqual(
                    np.linalg.norm(diff),
                    0,
                    delta=1,
                    msg="fast transform returned location {} but expected {} for point {}".format(
                        point.location, points_fast[point_id].location, point_id
                    ),
                )

            t_fast, t_ref = [np.mean(x) for x in zip(*timings)]
            self.assertLess(t_fast, t_ref)
            self.assertGreater(missing, 0)
