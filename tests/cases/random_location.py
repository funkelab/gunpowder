from .provider_test import ProviderTest
from gunpowder import *
import numpy as np


class TestSourceRandomLocation(BatchProvider):
    def __init__(self):
        self.roi = Roi((-200, -20, -20), (1000, 100, 100))
        self.voxel_size = Coordinate((20, 2, 2))

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

        data = np.zeros(
            request[ArrayKeys.RAW].roi.get_shape() / self.voxel_size)
        if request.array_specs[ArrayKeys.RAW].roi.contains((0, 0, 0)):
            data[:] = 1

        batch.arrays[ArrayKeys.RAW] = Array(
            data=data,
            spec=spec)

        return batch


class CustomRandomLocation(RandomLocation):

    # only accept random locations that contain (0, 0, 0)
    def accepts(self, request):
        return request.array_specs[ArrayKeys.RAW].roi.contains((0, 0, 0))


class TestRandomLocation(ProviderTest):

    def test_output(self):

        source = TestSourceRandomLocation()
        pipeline = (
            source
            + CustomRandomLocation()
        )

        with build(pipeline):

            for i in range(10):
                batch = pipeline.request_batch(
                    BatchRequest(
                        {
                            ArrayKeys.RAW: ArraySpec(
                                roi=Roi((0, 0, 0), (20, 20, 20)))
                        }))

                self.assertTrue(np.sum(batch.arrays[ArrayKeys.RAW].data) > 0)

                # Request the entire ROI from the source
                batch = pipeline.request_batch(
                    BatchRequest(
                        {
                            ArrayKeys.RAW: ArraySpec(roi=source.roi)
                        }
                    )
                )
