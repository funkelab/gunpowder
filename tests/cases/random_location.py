from .provider_test import ProviderTest
from gunpowder import (
    ArrayKeys,
    ArraySpec,
    Array,
    Roi,
    Coordinate,
    Batch,
    BatchRequest,
    BatchProvider,
    RandomLocation,
    build,
)
import numpy as np

class TestSourceRandomLocation(BatchProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_shape = (60, 60, 60)
        self.data_voxel_size = (20, 2, 2)
        x = np.linspace(-10, 49, 60).reshape((-1, 1, 1))
        self.data = x + x.transpose([1, 2, 0]) + x.transpose([2, 0, 1])

    def setup(self):

        self.provides(
            ArrayKeys.RAW,
            ArraySpec(
                roi=Roi((-200, -20, -20), (1000, 100, 100)),
                voxel_size=(20, 2, 2)))

    def provide(self, request):

        batch = Batch()

        spec = request[ArrayKeys.RAW].copy()
        spec.voxel_size = Coordinate((20, 2, 2))
        
        start = ((request[ArrayKeys.RAW].roi.get_begin() / self.data_voxel_size) + (10,10,10))
        end = ((request[ArrayKeys.RAW].roi.get_end() / self.data_voxel_size) + (10,10,10))
        data_slices = tuple(map(slice, start, end))

        data = self.data[data_slices]

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

        pipeline = (
            TestSourceRandomLocation() +
            CustomRandomLocation()
        )

        with build(pipeline):

            for i in range(10):
                batch = pipeline.request_batch(
                    BatchRequest(
                        {
                            ArrayKeys.RAW: ArraySpec(
                                roi=Roi((0, 0, 0), (20, 20, 20)))
                        }))

                self.assertTrue(0 in batch.arrays[ArrayKeys.RAW].data)

    def test_random_seed(self):
        pipeline = TestSourceRandomLocation() + CustomRandomLocation()

        with build(pipeline):
            seeded_sums = []
            unseeded_sums = []
            for i in range(100):
                batch_seeded = pipeline.request_batch(
                    BatchRequest(
                        {ArrayKeys.RAW: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20)))},
                        random_seed=10,
                    )
                )
                seeded_sums.append(batch_seeded[ArrayKeys.RAW].data.sum())
                batch_unseeded = pipeline.request_batch(
                    BatchRequest(
                        {ArrayKeys.RAW: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20)))},
                    )
                )
                unseeded_sums.append(batch_unseeded[ArrayKeys.RAW].data.sum())

            self.assertEqual(len(set(seeded_sums)), 1)
            self.assertGreater(len(set(unseeded_sums)), 1)

