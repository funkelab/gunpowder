from gunpowder import *
from gunpowder.ext import zarr, ZarrFile, NoSuchModule
from unittest import skipIf
import numpy as np

class ZarrWriteTestSource(BatchProvider):

    def setup(self):

        self.provides(
            raw_key,
            ArraySpec(
                roi=Roi((20000, 2000, 2000), (2000, 200, 200)),
                voxel_size=(20, 2, 2)))
        self.provides(
            gt_key,
            ArraySpec(
                roi=Roi((20100, 2010, 2010), (1800, 180, 180)),
                voxel_size=(20, 2, 2)))

    def provide(self, request):

        # print("ZarrWriteTestSource: Got request " + str(request))

        batch = Batch()

        # have the pixels encode their position
        for (array_key, spec) in request.array_specs.items():

            roi = spec.roi
            roi_voxel = roi // self.spec[array_key].voxel_size
            # print("ZarrWriteTestSource: Adding " + str(array_key))

            # the z,y,x coordinates of the ROI
            meshgrids = np.meshgrid(
                    range(roi_voxel.begin[0], roi_voxel.end[0]),
                    range(roi_voxel.begin[1], roi_voxel.end[1]),
                    range(roi_voxel.begin[2], roi_voxel.end[2]), indexing='ij')
            data = np.array(meshgrids)

            # print("Roi is: " + str(roi))

            spec = self.spec[array_key].copy()
            spec.roi = roi
            batch.arrays[array_key] = Array(
                    data,
                    spec)
        return batch

@skipIf(isinstance(zarr, NoSuchModule), 'zarr is not installed')
def test_read_write(self):

    raw_key = ArrayKey("RAW")
    gt_key = ArrayKey("GT")

    raw_data = np.array(np.meshgrid(
                    range(roi_voxel.begin[0], roi_voxel.end[0]),
                    range(roi_voxel.begin[1], roi_voxel.end[1]),
                    range(roi_voxel.begin[2], roi_voxel.end[2]), indexing='ij'))
    gt_data = 

    path = self.path_to('zarr_write_test.zarr')

    source = ZarrWriteTestSource()

    chunk_request = BatchRequest()
    chunk_request.add(raw_key, (400,30,34))
    chunk_request.add(gt_key, (200,10,14))

    pipeline = (
        source +
        ZarrWrite({
            raw_key: 'arrays/raw'
        },
        output_filename=path) +
        Scan(chunk_request))

    with build(pipeline):

        raw_spec    = pipeline.spec[raw_key]
        labels_spec = pipeline.spec[gt_key]

        full_request = BatchRequest({
                raw_key: raw_spec,
                gt_key: labels_spec
            }
        )

        batch = pipeline.request_batch(full_request)

    # assert that stored HDF dataset equals batch array

    with ZarrFile(path, mode='r') as f:

        ds = f['arrays/raw']

        batch_raw = batch.arrays[raw_key]
        stored_raw = ds[:]

        self.assertEqual(
            stored_raw.shape[-3:],
            batch_raw.spec.roi.shape//batch_raw.spec.voxel_size)
        self.assertEqual(tuple(ds.attrs['offset']), batch_raw.spec.roi.offset)
        self.assertEqual(tuple(ds.attrs['resolution']), batch_raw.spec.voxel_size)
        self.assertTrue((stored_raw == batch.arrays[raw_key].data).all())

