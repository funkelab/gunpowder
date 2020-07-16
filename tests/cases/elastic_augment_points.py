import unittest
from gunpowder import *
from gunpowder.points import GraphKeys, Graph, Node
from .provider_test import ProviderTest

import numpy as np
import math
from random import randint

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
            GraphSpec(
                roi=Roi((-100, -100, -100), (200, 200, 200))
            ))

        self.provides(
            ArrayKeys.TEST_LABELS,
            ArraySpec(
                roi=Roi((-100, -100, -100), (200, 200, 200)),
                voxel_size=Coordinate((4, 1, 1)),
                interpolatable=False
            ))

    def point_to_voxel(self, array_roi, location):

        # location is in world units, get it into voxels
        location = location/self.spec[ArrayKeys.TEST_LABELS].voxel_size

        # shift location relative to beginning of array roi
        location -= array_roi.get_begin()/self.spec[ArrayKeys.TEST_LABELS].voxel_size

        return tuple(
            slice(int(l-2), int(l+3))
            for l in location)

    def provide(self, request):

        batch = Batch()

        roi_points = request[GraphKeys.TEST_POINTS].roi
        roi_array = request[ArrayKeys.TEST_LABELS].roi
        roi_voxel = roi_array//self.spec[ArrayKeys.TEST_LABELS].voxel_size

        data = np.zeros(roi_voxel.get_shape(), dtype=np.uint32)
        data[:,::2] = 100

        for node in self.points:
            loc = self.point_to_voxel(roi_array, node.location)
            data[loc] = node.id

        spec = self.spec[ArrayKeys.TEST_LABELS].copy()
        spec.roi = roi_array
        batch.arrays[ArrayKeys.TEST_LABELS] = Array(
            data,
            spec=spec)

        points = []
        for node in self.points:
            if roi_points.contains(node.location):
                points.append(node)
        batch.graphs[GraphKeys.TEST_POINTS] = Graph(
            points,
            [],
            GraphSpec(roi=roi_points))

        return batch


class TestElasticAugment(ProviderTest):

    def test_3d_basics(self):

        test_labels = ArrayKey('TEST_LABELS')
        test_points = GraphKey('TEST_POINTS')
        test_raster = ArrayKey('TEST_RASTER')

        pipeline = (

            PointTestSource3D() +
            ElasticAugment(
                [10, 10, 10],
                [0.1, 0.1, 0.1],
                # [0, 0, 0], # no jitter
                [0, 2.0*math.pi]) + # rotate randomly
                # [math.pi/4, math.pi/4]) + # rotate by 45 deg
                # [0, 0]) + # no rotation
            RasterizePoints(
                test_points,
                test_raster,
                settings=RasterizationSettings(
                    radius=2,
                    mode='peak')) +
            Snapshot(
                {
                    test_labels: 'volumes/labels',
                    test_raster: 'volumes/raster'
                },
                dataset_dtypes={test_raster: np.float32},
                output_dir=self.path_to(),
                output_filename='elastic_augment_test{id}-{iteration}.hdf'
            )
        )

        for _ in range(5):

            with build(pipeline):

                request_roi = Roi(
                    (-20, -20, -20),
                    (40, 40, 40))

                request = BatchRequest()
                request[test_labels] = ArraySpec(roi=request_roi)
                request[test_points] = GraphSpec(roi=request_roi)
                request[test_raster] = ArraySpec(roi=request_roi)

                batch = pipeline.request_batch(request)
                labels = batch[test_labels]
                points = batch[test_points]

                # the point at (0, 0, 0) should not have moved
                data = {node.id: node for node in points.nodes}
                self.assertTrue(0 in data)

                labels_data_roi = (
                    labels.spec.roi -
                    labels.spec.roi.get_begin())/labels.spec.voxel_size

                # points should have moved together with the voxels
                for node in points.nodes:
                    loc = node.location - labels.spec.roi.get_begin()
                    loc = loc/labels.spec.voxel_size
                    loc = Coordinate(int(round(x)) for x in loc)
                    if labels_data_roi.contains(loc):
                        self.assertEqual(labels.data[loc], node.id)
