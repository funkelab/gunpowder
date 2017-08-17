import unittest
from gunpowder import *
from gunpowder.points import PointsTypes, Points

import numpy as np

class PointTestSource(BatchProvider):

    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def get_spec(self):
        spec = ProviderSpec()
        spec.points[PointsTypes.PRESYN] = Roi((0, 0, 0), (100, 100, 100))
        return spec

    def provide(self, request):
        batch = Batch()
        roi_points = request.points[PointsTypes.PRESYN]

        batch.points[PointsTypes.PRESYN] = Points(data={}, roi=roi_points, resolution=self.voxel_size)
        return batch

class VolumeTestSoure(BatchProvider):

    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def get_spec(self):
        spec = ProviderSpec()
        spec.volumes[VolumeTypes.GT_LABELS] = Roi((0, 0, 0), (100, 100, 100))
        return spec


    def provide(self, request):
        roi_volume = request.volumes[VolumeTypes.GT_LABELS]
        print roi_volume
        data = np.zeros(roi_volume.get_shape() / VolumeTypes.GT_LABELS.voxel_size)
        batch = Batch()
        batch.volumes[VolumeTypes.GT_LABELS] = Volume(data, roi=roi_volume)
        return batch


class TestMergeProvider(unittest.TestCase):

    def test_merge_basics(self):
        voxel_size = tuple([1, 1, 1])
        pointssource = PointTestSource(voxel_size)
        volumesource = VolumeTestSoure(voxel_size)
        pipeline = tuple([pointssource, volumesource]) + MergeProvider() + RandomLocation()
        window_request = Coordinate((50, 50, 50))
        with build(pipeline):
            # Check basic merging.
            request = BatchRequest()
            request.add_points_request((PointsTypes.PRESYN), window_request)
            request.add_volume_request((VolumeTypes.GT_LABELS), window_request)
            batch_res = pipeline.request_batch(request)
            self.assertTrue(VolumeTypes.GT_LABELS in batch_res.volumes)
            self.assertTrue(PointsTypes.PRESYN in batch_res.points)

            # Check that request of only one source also works.
            request = BatchRequest()
            request.add_points_request((PointsTypes.PRESYN), window_request)
            batch_res = pipeline.request_batch(request)
            self.assertFalse(VolumeTypes.GT_LABELS in batch_res.volumes)
            self.assertTrue(PointsTypes.PRESYN in batch_res.points)

        # Check that it fails, when having two sources that provide the same type.
        volumesource2 = VolumeTestSoure(voxel_size)
        pipeline_fail = tuple([volumesource, volumesource2]) + MergeProvider() + RandomLocation()
        with self.assertRaises(AssertionError):
            with build(pipeline_fail):
                pass


