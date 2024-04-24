import numpy as np
import pytest

from gunpowder import (
    ArrayKey,
    ArraySpec,
    BatchRequest,
    Hdf5Source,
    Roi,
    ZarrSource,
    build,
)
from gunpowder.ext import NoSuchModule, ZarrFile, h5py, zarr

extension = None
SourceUnderTest = None


def open_zarr(path):
    return ZarrFile(path, mode="w")


def open_hdf(path):
    return h5py.File(path, "w")


open_writable_file_func = {
    "hdf": open_hdf,
    "zarr": open_zarr,
}
source_node = {
    "hdf": Hdf5Source,
    "zarr": ZarrSource,
}


def create_dataset(data_file, key, data, chunks=None, **kwargs):
    chunks = chunks or data.shape
    d = data_file.create_dataset(key, shape=data.shape, dtype=data.dtype, chunks=chunks)
    d[:] = data
    for key, value in kwargs.items():
        d.attrs[key] = value


@pytest.mark.parametrize(
    "extension",
    [
        "hdf",
        pytest.param(
            "zarr",
            marks=pytest.mark.skipif(
                isinstance(zarr, NoSuchModule), reason="zarr is not installed"
            ),
        ),
    ],
)
def test_output_2d(extension, tmpdir):
    path = tmpdir / f"test_{extension}_source.{extension}"

    with open_writable_file_func[extension](path) as f:
        create_dataset(f, "raw", np.zeros((100, 100), dtype=np.float32))
        create_dataset(
            f, "raw_low", np.zeros((10, 10), dtype=np.float32), resolution=(10, 10)
        )
        create_dataset(f, "seg", np.ones((100, 100), dtype=np.uint64))

    # read arrays
    raw = ArrayKey("RAW")
    raw_low = ArrayKey("RAW_LOW")
    seg = ArrayKey("SEG")
    source = source_node[extension](path, {raw: "raw", raw_low: "raw_low", seg: "seg"})

    with build(source):
        batch = source.request_batch(
            BatchRequest(
                {
                    raw: ArraySpec(roi=Roi((0, 0), (100, 100))),
                    raw_low: ArraySpec(roi=Roi((0, 0), (100, 100))),
                    seg: ArraySpec(roi=Roi((0, 0), (100, 100))),
                }
            )
        )

        assert batch.arrays[raw].spec.interpolatable
        assert batch.arrays[raw_low].spec.interpolatable
        assert not (batch.arrays[seg].spec.interpolatable)


@pytest.mark.parametrize(
    "extension",
    [
        "hdf",
        pytest.param(
            "zarr",
            marks=pytest.mark.skipif(
                isinstance(zarr, NoSuchModule), reason="zarr is not installed"
            ),
        ),
    ],
)
def test_output_3d(extension, tmpdir):
    path = tmpdir / f"test_{extension}_source.{extension}"

    # create a test file
    with open_writable_file_func[extension](path) as f:
        create_dataset(f, "raw", np.zeros((100, 100, 100), dtype=np.float32))
        create_dataset(
            f,
            "raw_low",
            np.zeros((10, 10, 10), dtype=np.float32),
            resolution=(10, 10, 10),
        )
        create_dataset(f, "seg", np.ones((100, 100, 100), dtype=np.uint64))

    # read arrays
    raw = ArrayKey("RAW")
    raw_low = ArrayKey("RAW_LOW")
    seg = ArrayKey("SEG")
    source = source_node[extension](path, {raw: "raw", raw_low: "raw_low", seg: "seg"})

    with build(source):
        batch = source.request_batch(
            BatchRequest(
                {
                    raw: ArraySpec(roi=Roi((0, 0, 0), (100, 100, 100))),
                    raw_low: ArraySpec(roi=Roi((0, 0, 0), (100, 100, 100))),
                    seg: ArraySpec(roi=Roi((0, 0, 0), (100, 100, 100))),
                }
            )
        )

        assert batch.arrays[raw].spec.interpolatable
        assert batch.arrays[raw_low].spec.interpolatable
        assert not (batch.arrays[seg].spec.interpolatable)
