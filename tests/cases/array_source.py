import numpy as np
from funlib.geometry import Roi
from funlib.persistence import prepare_ds

from gunpowder import ArrayKey, ArraySpec, BatchRequest, build
from gunpowder.nodes import ArraySource


def test_array_source(tmpdir):
    array = prepare_ds(
        tmpdir / "data.zarr",
        shape=(100, 102, 108),
        offset=(100, 50, 0),
        voxel_size=(1, 2, 3),
        dtype="uint8",
    )
    array[:] = np.arange(100 * 102 * 108).reshape((100, 102, 108)) % 255

    key = ArrayKey("TEST")

    source = ArraySource(key=key, array=array)

    with build(source):
        request = BatchRequest()

        roi = Roi((100, 100, 102), (30, 30, 30))
        request[key] = ArraySpec(roi)

        assert np.array_equal(source.request_batch(request)[key].data, array[roi])
