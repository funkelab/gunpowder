import numpy as np

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    BatchRequest,
    IntensityAugment,
    Roi,
    build,
)

from .helper_sources import ArraySource


def test_shift():
    raw_key = ArrayKey("RAW")
    raw_spec = ArraySpec(
        roi=Roi((0, 0, 0), (10, 10, 10)), voxel_size=(1, 1, 1), dtype=np.float32
    )
    raw_data = np.zeros(raw_spec.roi.shape / raw_spec.voxel_size, dtype=np.float32)
    raw_array = Array(raw_data, raw_spec)

    pipeline = ArraySource(raw_key, raw_array) + IntensityAugment(
        raw_key, scale_min=0, scale_max=0, shift_min=0.5, shift_max=0.5
    )

    request = BatchRequest()
    request.add(raw_key, (10, 10, 10))

    with build(pipeline):
        batch = pipeline.request_batch(request)

        x = batch.arrays[raw_key].data
        assert np.isclose(x.min(), 0.5)
        assert np.isclose(x.max(), 0.5)
