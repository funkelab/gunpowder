import logging
import numpy as np

from .provider_test import ProviderTest
from gunpowder import (
    RandomLocation,
    BatchProvider,
    Roi,
    Coordinate,
    ArrayKey,
    ArrayKeys,
    ArraySpec,
    Batch,
    Array,
    BatchRequest,
    build,
    MergeProvider,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TestSourceRandomLocation(BatchProvider):
    def __init__(self, array, offset=(-200, -20, -20)):
        self.array = array
        self.roi = Roi(offset, (1000, 100, 100))
        self.voxel_size = Coordinate((20, 2, 2))

    def setup(self):
        self.provides(
            self.array,
            ArraySpec(
                roi=self.roi,
                voxel_size=self.voxel_size))

    def provide(self, request):

        batch = Batch()

        spec = request[self.array].copy()
        spec.voxel_size = self.voxel_size

        data = np.zeros(request[self.array].roi.get_shape() / self.voxel_size)
        if request.array_specs[self.array].roi.contains((0, 0, 0)):
            data[:] = 1

        batch.arrays[self.array] = Array(data=data, spec=spec)

        return batch


class CustomRandomLocation(RandomLocation):

    # only accept random locations that contain (0, 0, 0)
    def accepts(self, request):
        return request.array_specs[ArrayKeys.RAW].roi.contains((0, 0, 0))


class TestRandomLocation(ProviderTest):
    def test_output(self):

        raw = ArrayKeys.RAW
        source = TestSourceRandomLocation(raw)
        pipeline = source + CustomRandomLocation()

        with build(pipeline):

            for i in range(10):
                batch = pipeline.request_batch(
                    BatchRequest(
                        {raw: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20)))}
                    )
                )

                self.assertTrue(np.sum(batch.arrays[raw].data) > 0)

                # Request a ROI with the same shape as the entire ROI
                full_roi = Roi((0, 0, 0), source.roi.get_shape())
                batch = pipeline.request_batch(
                    BatchRequest({raw: ArraySpec(roi=full_roi)})
                )

                logger.debug(
                    f'batch shape {batch[raw].data.shape}')
                logger.debug(
                    f'full_roi shape {full_roi.get_shape() // source.voxel_size}')
                self.assertTrue(
                    batch[raw].data.shape == full_roi.get_shape() //
                    source.voxel_size)

    def test_output_zero_offset(self):

        raw = ArrayKeys.RAW
        source = TestSourceRandomLocation(raw, offset=(0, 0, 0))
        pipeline = source + CustomRandomLocation()

        with build(pipeline):

            for i in range(10):
                batch = pipeline.request_batch(
                    BatchRequest(
                        {raw: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20)))}
                    )
                )

                self.assertTrue(np.sum(batch.arrays[raw].data) > 0)

                # Request a ROI with the same shape as the entire ROI
                full_roi = Roi((0, 0, 0), source.roi.get_shape())
                batch = pipeline.request_batch(
                    BatchRequest({raw: ArraySpec(roi=full_roi)})
                )

                logger.debug(
                    f'batch shape {batch[raw].data.shape}')
                logger.debug(
                    f'full_roi shape {full_roi.get_shape() // source.voxel_size}')
                self.assertTrue(
                    batch[raw].data.shape == full_roi.get_shape() //
                    source.voxel_size)

    def test_impossible(self):
        a = ArrayKey("A")
        b = ArrayKey("B")
        source_a = TestSourceRandomLocation(a)
        source_b = TestSourceRandomLocation(b)

        pipeline = (source_a, source_b) + \
            MergeProvider() + CustomRandomLocation()

        with build(pipeline):
            with self.assertRaises(AssertionError):
                batch = pipeline.request_batch(
                    BatchRequest(
                        {
                            a: ArraySpec(roi=Roi((0, 0, 0), (200, 20, 20))),
                            b: ArraySpec(
                                roi=Roi(
                                    (220, 22, 22), (1000, 100, 100))),
                        }
                    )
                )


if __name__ == '__main__':
    pass
