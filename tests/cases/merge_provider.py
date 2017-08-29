import unittest
from gunpowder import *
from gunpowder.points import PointsTypes, Points

import numpy as np

class PointTestSource(BatchProvider):

    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def setup(self):

        self.provides(
            PointsTypes.PRESYN,
            PointsSpec(roi=Roi((0, 0, 0), (100, 100, 100))))

    def provide(self, request):
        batch = Batch()
        roi_points = request[PointsTypes.PRESYN].roi

        batch.points[PointsTypes.PRESYN] = Points(
            {},
            PointsSpec(roi=roi_points))
        return batch

class VolumeTestSoure(BatchProvider):

    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def setup(self):

        self.provides(
            VolumeTypes.GT_LABELS,
            VolumeSpec(
                roi=Roi((0, 0, 0), (100, 100, 100)),
                voxel_size=self.voxel_size))

    def provide(self, request):
        roi_volume = request[VolumeTypes.GT_LABELS].roi
        print roi_volume
        data = np.zeros(
            roi_volume.get_shape() /
            self.spec[VolumeTypes.GT_LABELS].voxel_size)
        batch = Batch()
        spec = self.spec[VolumeTypes.GT_LABELS].copy()
        spec.roi = roi_volume
        batch.volumes[VolumeTypes.GT_LABELS] = Volume(
            data,
            spec)
        return batch


class TestMergeProvider(unittest.TestCase):

    def test_merge_basics(self):
        voxel_size = (1, 1, 1)
        pointssource = PointTestSource(voxel_size)
        volumesource = VolumeTestSoure(voxel_size)
        pipeline = (pointssource, volumesource) + MergeProvider() + RandomLocation()
        window_request = Coordinate((50, 50, 50))
        with build(pipeline):
            # Check basic merging.
            request = BatchRequest()
            request.add((PointsTypes.PRESYN), window_request)
            request.add((VolumeTypes.GT_LABELS), window_request)
            batch_res = pipeline.request_batch(request)
            self.assertTrue(VolumeTypes.GT_LABELS in batch_res.volumes)
            self.assertTrue(PointsTypes.PRESYN in batch_res.points)

            # Check that request of only one source also works.
            request = BatchRequest()
            request.add((PointsTypes.PRESYN), window_request)
            batch_res = pipeline.request_batch(request)
            self.assertFalse(VolumeTypes.GT_LABELS in batch_res.volumes)
            self.assertTrue(PointsTypes.PRESYN in batch_res.points)

        # Check that it fails, when having two sources that provide the same type.
        volumesource2 = VolumeTestSoure(voxel_size)
        pipeline_fail = (volumesource, volumesource2) + MergeProvider() + RandomLocation()
        with self.assertRaises(AssertionError):
            with build(pipeline_fail):
                pass


