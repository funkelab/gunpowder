import unittest
from gunpowder import *
from gunpowder.points import PointsKeys, Points

import numpy as np

class PointTestSource(BatchProvider):

    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def setup(self):

        self.provides(
            PointsKeys.PRESYN,
            PointsSpec(roi=Roi((0, 0, 0), (100, 100, 100))))

    def provide(self, request):
        batch = Batch()
        roi_points = request[PointsKeys.PRESYN].roi

        batch.points[PointsKeys.PRESYN] = Points(
            {},
            PointsSpec(roi=roi_points))
        return batch

class ArrayTestSoure(BatchProvider):

    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def setup(self):

        self.provides(
            ArrayKeys.GT_LABELS,
            ArraySpec(
                roi=Roi((0, 0, 0), (100, 100, 100)),
                voxel_size=self.voxel_size))

    def provide(self, request):
        roi_array = request[ArrayKeys.GT_LABELS].roi
        data = np.zeros(
            roi_array.get_shape() /
            self.spec[ArrayKeys.GT_LABELS].voxel_size)
        batch = Batch()
        spec = self.spec[ArrayKeys.GT_LABELS].copy()
        spec.roi = roi_array
        batch.arrays[ArrayKeys.GT_LABELS] = Array(
            data,
            spec)
        return batch


class TestMergeProvider(unittest.TestCase):

    def test_merge_basics(self):
        voxel_size = (1, 1, 1)
        PointsKey('PRESYN')
        pointssource = PointTestSource(voxel_size)
        arraysource = ArrayTestSoure(voxel_size)
        pipeline = (pointssource, arraysource) + MergeProvider() + RandomLocation()
        window_request = Coordinate((50, 50, 50))
        with build(pipeline):
            # Check basic merging.
            request = BatchRequest()
            request.add((PointsKeys.PRESYN), window_request)
            request.add((ArrayKeys.GT_LABELS), window_request)
            batch_res = pipeline.request_batch(request)
            self.assertTrue(ArrayKeys.GT_LABELS in batch_res.arrays)
            self.assertTrue(PointsKeys.PRESYN in batch_res.points)

            # Check that request of only one source also works.
            request = BatchRequest()
            request.add((PointsKeys.PRESYN), window_request)
            batch_res = pipeline.request_batch(request)
            self.assertFalse(ArrayKeys.GT_LABELS in batch_res.arrays)
            self.assertTrue(PointsKeys.PRESYN in batch_res.points)

        # Check that it fails, when having two sources that provide the same type.
        arraysource2 = ArrayTestSoure(voxel_size)
        pipeline_fail = (arraysource, arraysource2) + MergeProvider() + RandomLocation()
        with self.assertRaises(AssertionError):
            with build(pipeline_fail):
                pass


