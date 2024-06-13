import numpy as np
import pytest

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    Batch,
    BatchProvider,
    BatchRequest,
    Coordinate,
    MergeProvider,
    RandomLocation,
    Roi,
    build,
)
from gunpowder.pipeline import PipelineRequestError


class ExampleSourceRandomLocation(BatchProvider):
    def __init__(self, array):
        self.array = array
        self.roi = Roi((-200, -20, -20), (1000, 100, 100))
        self.data_shape = (60, 60, 60)
        self.voxel_size = Coordinate(20, 2, 2)
        x = np.linspace(-10, 49, 60).reshape((-1, 1, 1))
        self.data = x + x.transpose([1, 2, 0]) + x.transpose([2, 0, 1])

    def setup(self):
        self.provides(self.array, ArraySpec(roi=self.roi, voxel_size=self.voxel_size))

    def provide(self, request):
        batch = Batch()

        spec = request[self.array].copy()
        spec.voxel_size = self.voxel_size

        start = (request[self.array].roi.begin / self.voxel_size) + 10
        end = (request[self.array].roi.end / self.voxel_size) + 10
        data_slices = tuple(map(slice, start, end))

        data = self.data[data_slices]

        batch.arrays[self.array] = Array(data=data, spec=spec)

        return batch


class CustomRandomLocation(RandomLocation):
    def __init__(self, array, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.array = array

    # only accept random locations that contain (0, 0, 0)
    def accepts(self, request):
        return request.array_specs[self.array].roi.contains((0, 0, 0))


def test_output():
    a = ArrayKey("A")
    b = ArrayKey("B")
    random_shift_key = ArrayKey("RANDOM_SHIFT")

    pipeline = (
        (ExampleSourceRandomLocation(a), ExampleSourceRandomLocation(b))
        + MergeProvider()
        + CustomRandomLocation(a, random_shift_key=random_shift_key)
    )
    pipeline_no_random = (
        ExampleSourceRandomLocation(a),
        ExampleSourceRandomLocation(b),
    ) + MergeProvider()

    with build(pipeline), build(pipeline_no_random):
        sums = set()
        for i in range(10):
            batch = pipeline.request_batch(
                BatchRequest(
                    {
                        a: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20))),
                        b: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20))),
                        random_shift_key: ArraySpec(nonspatial=True),
                    }
                )
            )

            assert 0 in batch.arrays[a].data
            assert 0 in batch.arrays[b].data

            # check that we can repeat this request without the random location
            batch_no_random = pipeline_no_random.request_batch(
                BatchRequest(
                    {
                        a: ArraySpec(
                            roi=Roi(batch[random_shift_key].data, (20, 20, 20))
                        ),
                        b: ArraySpec(
                            roi=Roi(batch[random_shift_key].data, (20, 20, 20))
                        ),
                    }
                )
            )

            assert batch_no_random.arrays[a].data.sum() == batch.arrays[a].data.sum()

            sums.add(batch[a].data.sum())

            # Request a ROI with the same shape as the entire ROI
            full_roi_a = Roi((0, 0, 0), ExampleSourceRandomLocation(a).roi.shape)
            full_roi_b = Roi((0, 0, 0), ExampleSourceRandomLocation(b).roi.shape)
            batch = pipeline.request_batch(
                BatchRequest(
                    {a: ArraySpec(roi=full_roi_a), b: ArraySpec(roi=full_roi_b)}
                )
            )
        assert len(sums) > 1


def test_output():
    a = ArrayKey("A")
    b = ArrayKey("B")
    source_a = ExampleSourceRandomLocation(a)
    source_b = ExampleSourceRandomLocation(b)

    pipeline = (source_a, source_b) + MergeProvider() + CustomRandomLocation(a)

    with build(pipeline):
        for i in range(10):
            batch = pipeline.request_batch(
                BatchRequest(
                    {
                        a: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20))),
                        b: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20))),
                    }
                )
            )

            assert 0 in batch.arrays[a].data
            assert 0 in batch.arrays[b].data

            # Request a ROI with the same shape as the entire ROI
            full_roi_a = Roi((0, 0, 0), source_a.roi.shape)
            full_roi_b = Roi((0, 0, 0), source_b.roi.shape)
            batch = pipeline.request_batch(
                BatchRequest(
                    {a: ArraySpec(roi=full_roi_a), b: ArraySpec(roi=full_roi_b)}
                )
            )


def test_random_seed():
    raw = ArrayKey("RAW")
    pipeline = ExampleSourceRandomLocation(raw) + CustomRandomLocation(raw)

    with build(pipeline):
        seeded_sums = []
        unseeded_sums = []
        for i in range(10):
            batch_seeded = pipeline.request_batch(
                BatchRequest(
                    {raw: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20)))},
                    random_seed=10,
                )
            )
            seeded_sums.append(batch_seeded[raw].data.sum())
            batch_unseeded = pipeline.request_batch(
                BatchRequest({raw: ArraySpec(roi=Roi((0, 0, 0), (20, 20, 20)))})
            )
            unseeded_sums.append(batch_unseeded[raw].data.sum())

        assert len(set(seeded_sums)) == 1
        assert len(set(unseeded_sums)) > 1


def test_impossible():
    a = ArrayKey("A")
    b = ArrayKey("B")
    null_key = ArrayKey("NULL")
    source_a = ExampleSourceRandomLocation(a)
    source_b = ExampleSourceRandomLocation(b)

    pipeline = (source_a, source_b) + MergeProvider() + CustomRandomLocation(null_key)

    with build(pipeline):
        with pytest.raises(PipelineRequestError):
            batch = pipeline.request_batch(
                BatchRequest(
                    {
                        a: ArraySpec(roi=Roi((0, 0, 0), (200, 20, 20))),
                        b: ArraySpec(roi=Roi((1000, 100, 100), (220, 22, 22))),
                    }
                )
            )
