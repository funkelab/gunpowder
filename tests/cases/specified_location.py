# from .provider_test import ProviderTest, ExampleSource
from gunpowder import (BatchProvider, ArrayKeys, ArraySpec, Roi, Batch,
                       Coordinate, SpecifiedLocation, build,
                       BatchRequest, Array, ArrayKey)
import numpy as np
import unittest


class ExampleSourceSpecifiedLocation(BatchProvider):
    def __init__(self, roi, voxel_size):
        self.voxel_size = Coordinate(voxel_size)
        self.roi = roi
        size = self.roi.get_shape() / self.voxel_size
        self.data = np.arange(np.prod(size)).reshape(size)

    def setup(self):
        self.provides(
            ArrayKeys.RAW,
            ArraySpec(
                roi=self.roi,
                voxel_size=self.voxel_size))

    def provide(self, request):
        batch = Batch()

        spec = request[ArrayKeys.RAW].copy()
        spec.voxel_size = self.voxel_size
        size = spec.roi.get_shape() / spec.voxel_size
        offset = spec.roi.get_offset() / spec.voxel_size
        slce = tuple(slice(o, o + s) for o, s in zip(offset, size))

        batch.arrays[ArrayKeys.RAW] = Array(
            data=self.data[slce],
            spec=spec)

        return batch


class TestSpecifiedLocation(unittest.TestCase):

    def setUp(self):
        ArrayKey('RAW')

    def test_simple(self):

        locations = [
                [0, 0, 0],
                [100, 100, 100],
                [91, 20, 20],
                [42, 24, 57]
                ]

        pipeline = (
            ExampleSourceSpecifiedLocation(
                roi=Roi((0, 0, 0), (100, 100, 100)),
                voxel_size=(1, 1, 1)) +
            SpecifiedLocation(
                locations,
                choose_randomly=False,
                extra_data=None,
                jitter=None)
        )

        with build(pipeline):

            batch = pipeline.request_batch(
                BatchRequest(
                    {
                        ArrayKeys.RAW: ArraySpec(
                            roi=Roi((0, 0, 0), (20, 20, 20)))
                    }))
            # first three locations are skipped
            # fourth should start at [32, 14, 47] of self.data
            self.assertEqual(batch.arrays[ArrayKeys.RAW].data[0, 0, 0], 321447)

    def test_voxel_size(self):

        locations = [
                [0, 0, 0],
                [91, 20, 20],
                [42, 24, 57]
                ]

        pipeline = (
            ExampleSourceSpecifiedLocation(
                roi=Roi((0, 0, 0), (100, 100, 100)),
                voxel_size=(5, 2, 2)) +
            SpecifiedLocation(
                locations,
                choose_randomly=False,
                extra_data=None,
                jitter=None)
        )

        with build(pipeline):

            batch = pipeline.request_batch(
                BatchRequest(
                    {
                        ArrayKeys.RAW: ArraySpec(
                            roi=Roi((0, 0, 0), (20, 20, 20)))
                    }))
            # first locations is skipped
            # second should start at [80/5, 10/2, 10/2] = [16, 5, 5]
            self.assertEqual(batch.arrays[ArrayKeys.RAW].data[0, 0, 0], 40255)

            batch = pipeline.request_batch(
                BatchRequest(
                    {
                        ArrayKeys.RAW: ArraySpec(
                            roi=Roi((0, 0, 0), (20, 20, 20)))
                    }))
            # third should start at [30/5, 14/2, 48/2] = [6, 7, 23]
            self.assertEqual(batch.arrays[ArrayKeys.RAW].data[0, 0, 0], 15374)

    def test_jitter_and_random(self):

        locations = [
                [0, 0, 0],
                [91, 20, 20],
                [42, 24, 57]
                ]

        pipeline = (
            ExampleSourceSpecifiedLocation(
                roi=Roi((0, 0, 0), (100, 100, 100)),
                voxel_size=(5, 2, 2)) +
            SpecifiedLocation(
                locations,
                choose_randomly=True,
                extra_data=None,
                jitter=(5, 5, 5))
        )

        with build(pipeline):

            batch = pipeline.request_batch(
                BatchRequest(
                    {
                        ArrayKeys.RAW: ArraySpec(
                            roi=Roi((0, 0, 0), (20, 20, 20)))
                    }))
            # Unclear what result should be, so no errors means passing
            self.assertTrue(batch.arrays[ArrayKeys.RAW].data[0, 0, 0] > 0)
