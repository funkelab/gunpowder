import numpy as np

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    Batch,
    BatchProvider,
    BatchRequest,
    Hdf5Write,
    Roi,
    Scan,
    build,
)
from gunpowder.ext import h5py


class Hdf5WriteTestSource(BatchProvider):
    def __init__(self, raw_key, labels_key):
        self.raw_key = raw_key
        self.labels_key = labels_key

    def setup(self):
        self.provides(
            self.raw_key,
            ArraySpec(
                roi=Roi((20000, 2000, 2000), (2000, 200, 200)), voxel_size=(20, 2, 2)
            ),
        )
        self.provides(
            self.labels_key,
            ArraySpec(
                roi=Roi((20100, 2010, 2010), (1800, 180, 180)), voxel_size=(20, 2, 2)
            ),
        )

    def provide(self, request):
        # print("Hdf5WriteTestSource: Got request " + str(request))

        batch = Batch()

        # have the pixels encode their position
        for array_key, spec in request.array_specs.items():
            roi = spec.roi
            roi_voxel = roi // self.spec[array_key].voxel_size
            # print("Hdf5WriteTestSource: Adding " + str(array_key))

            # the z,y,x coordinates of the ROI
            meshgrids = np.meshgrid(
                range(roi_voxel.begin[0], roi_voxel.end[0]),
                range(roi_voxel.begin[1], roi_voxel.end[1]),
                range(roi_voxel.begin[2], roi_voxel.end[2]),
                indexing="ij",
            )
            data = np.array(meshgrids)

            # print("Roi is: " + str(roi))

            spec = self.spec[array_key].copy()
            spec.roi = roi
            batch.arrays[array_key] = Array(data, spec)
        return batch


def test_output(tmpdir):
    path = tmpdir / "hdf5_write_test.hdf"

    raw_key = ArrayKey("RAW")
    labels_key = ArrayKey("LABELS")

    source = Hdf5WriteTestSource(raw_key, labels_key)

    chunk_request = BatchRequest()
    chunk_request.add(raw_key, (400, 30, 34))
    chunk_request.add(labels_key, (200, 10, 14))

    pipeline = (
        source
        + Hdf5Write({raw_key: "arrays/raw"}, output_filename=path)
        + Scan(chunk_request)
    )

    with build(pipeline):
        raw_spec = pipeline.spec[raw_key]
        labels_spec = pipeline.spec[labels_key]

        full_request = BatchRequest({raw_key: raw_spec, labels_key: labels_spec})

        batch = pipeline.request_batch(full_request)

    # assert that stored HDF dataset equals batch array

    with h5py.File(path, "r") as f:
        ds = f["arrays/raw"]

        batch_raw = batch.arrays[raw_key]
        stored_raw = np.array(ds)

        assert (
            stored_raw.shape[-3:]
            == batch_raw.spec.roi.shape // batch_raw.spec.voxel_size
        )
        assert tuple(ds.attrs["offset"]) == batch_raw.spec.roi.offset
        assert tuple(ds.attrs["resolution"]) == batch_raw.spec.voxel_size
        assert (stored_raw == batch.arrays[raw_key].data).all()
