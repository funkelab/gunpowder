import numpy as np

from gunpowder import (
    ArrayKey,
    ArraySpec,
    BatchRequest,
    Hdf5Source,
    Roi,
    build,
)
from gunpowder.ext import h5py


def create_dataset(data_file, key, data, chunks=None, **kwargs):
    chunks = chunks or data.shape
    d = data_file.create_dataset(key, shape=data.shape, dtype=data.dtype, chunks=chunks)
    d[:] = data
    for key, value in kwargs.items():
        d.attrs[key] = value


def test_output_2d(tmp_path):
    path = tmp_path / "test_hdf_source.hdf"

    with h5py.File(path, "w") as f:
        create_dataset(f, "raw", np.zeros((100, 100), dtype=np.float32))
        create_dataset(
            f, "raw_low", np.zeros((10, 10), dtype=np.float32), resolution=(10, 10)
        )
        create_dataset(f, "seg", np.ones((100, 100), dtype=np.uint64))

    # read arrays
    raw = ArrayKey("RAW")
    raw_low = ArrayKey("RAW_LOW")
    seg = ArrayKey("SEG")
    source = Hdf5Source(path, {raw: "raw", raw_low: "raw_low", seg: "seg"})

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


def test_output_3d(tmp_path):
    path = tmp_path / "test_hdf_source.hdf"

    # create a test file
    with h5py.File(path, "w") as f:
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
    source = Hdf5Source(path, {raw: "raw", raw_low: "raw_low", seg: "seg"})

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
