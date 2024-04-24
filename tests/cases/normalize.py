import numpy as np

from gunpowder import Array, ArrayKey, ArraySpec, BatchRequest, Normalize, Roi, build

from .helper_sources import ArraySource


def test_output():
    raw_key = ArrayKey("RAW")
    raw_spec = ArraySpec(
        roi=Roi((0, 0, 0), (10, 10, 10)), voxel_size=(1, 1, 1), dtype=np.uint8
    )
    raw_data = np.zeros(raw_spec.roi.shape / raw_spec.voxel_size, dtype=np.uint8) + 128
    raw_array = Array(raw_data, raw_spec)
    pipeline = ArraySource(raw_key, raw_array) + Normalize(raw_key)

    request = BatchRequest()
    request.add(raw_key, (10, 10, 10))

    with build(pipeline):
        batch = pipeline.request_batch(request)

        raw = batch.arrays[raw_key]
        assert raw.data.min() >= 0
        assert raw.data.max() <= 1
