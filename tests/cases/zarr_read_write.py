import numpy as np
import pytest

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    BatchRequest,
    Coordinate,
    MergeProvider,
    Roi,
    Scan,
    ZarrSource,
    ZarrWrite,
    build,
)
from gunpowder.ext import NoSuchModule, zarr

from .helper_sources import ArraySource


@pytest.mark.skipif(isinstance(zarr, NoSuchModule), reason="zarr is not installed")
@pytest.mark.parametrize(
    "zarr_store_func",
    [
        "tmp_path / 'zarr_write_test.zarr'",
        "tmp_path / 'zarr_write_test.n5'",
        "tmp_path / 'zarr_write_test.hdf'",
        "zarr.DirectoryStore(f'{tmp_path}/array.zarr')",
        "zarr.storage.TempStore(dir=tmp_path)",
    ],
)
def test_read_write(tmp_path, zarr_store_func):
    zarr_store = eval(zarr_store_func)
    raw_key = ArrayKey("RAW")
    gt_key = ArrayKey("GT")

    roi_raw = Roi((20000, 2000, 2000), (2000, 200, 200))
    roi_gt = Roi((20100, 2010, 2010), (1800, 180, 180))
    voxel_size = Coordinate(20, 2, 2)

    raw_data = np.array(
        np.meshgrid(
            range((roi_raw / voxel_size).begin[0], (roi_raw / voxel_size).end[0]),
            range((roi_raw / voxel_size).begin[1], (roi_raw / voxel_size).end[1]),
            range((roi_raw / voxel_size).begin[2], (roi_raw / voxel_size).end[2]),
            indexing="ij",
        )
    )
    gt_data = np.array(
        np.meshgrid(
            range((roi_gt / voxel_size).begin[0], (roi_gt / voxel_size).end[0]),
            range((roi_gt / voxel_size).begin[1], (roi_gt / voxel_size).end[1]),
            range((roi_gt / voxel_size).begin[2], (roi_gt / voxel_size).end[2]),
            indexing="ij",
        )
    )

    raw_array = Array(raw_data, ArraySpec(roi_raw, voxel_size))
    gt_array = Array(gt_data, ArraySpec(roi_gt, voxel_size))

    source = (
        ArraySource(raw_key, raw_array),
        ArraySource(gt_key, gt_array),
    ) + MergeProvider()

    chunk_request = BatchRequest()
    chunk_request.add(raw_key, (800, 80, 38))
    chunk_request.add(gt_key, (600, 60, 18))

    pipeline = (
        source
        + ZarrWrite({raw_key: "arrays/raw"}, store=zarr_store)
        + Scan(chunk_request)
    )

    with build(pipeline):
        raw_spec = pipeline.spec[raw_key]
        labels_spec = pipeline.spec[gt_key]

        full_request = BatchRequest({raw_key: raw_spec, gt_key: labels_spec})

        batch = pipeline.request_batch(full_request)

    # assert that stored HDF dataset equals batch array
    read_pipeline = ZarrSource(zarr_store, datasets={raw_key: "arrays/raw"})
    full_request = BatchRequest({raw_key: full_request[raw_key]})

    with build(read_pipeline):
        full_batch = read_pipeline.request_batch(full_request)

        assert (
            raw_data.shape[-3:]
            == full_batch[raw_key].spec.roi.shape // full_batch[raw_key].spec.voxel_size
        )
        assert roi_raw.offset == full_batch[raw_key].spec.roi.offset
        assert voxel_size == full_batch[raw_key].spec.voxel_size
        assert (raw_data == batch.arrays[raw_key].data).all()


def test_old_api(tmp_path):
    raw_key = ArrayKey("RAW")

    ZarrWrite({raw_key: "arrays/raw"}, tmp_path, "data.zarr")
    ZarrWrite({raw_key: "arrays/raw"}, output_dir=tmp_path, output_filename="data.zarr")

    ZarrSource(filename=f"{tmp_path}/data.zarr", datasets={raw_key: "arrays/raw"})
    ZarrSource(datasets=f"{tmp_path}/data.zarr", filename={raw_key: "arrays/raw"})
    ZarrSource(f"{tmp_path}/data.zarr", {raw_key: "arrays/raw"})
