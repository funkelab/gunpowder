from .provider_test import ProviderTest
from gunpowder import *
from gunpowder.ext import zarr, ZarrFile, NoSuchModule
from unittest import skipIf
import numpy as np

class ZarrWriteTestSource(BatchProvider):

    def setup(self):

        self.provides(
            ArrayKeys.RAW,
            ArraySpec(
                roi=Roi((20000, 2000, 2000), (2000, 200, 200)),
                voxel_size=(20, 2, 2)))
        self.provides(
            ArrayKeys.GT_LABELS,
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
                    range(roi_voxel.get_begin()[0], roi_voxel.get_end()[0]),
                    range(roi_voxel.get_begin()[1], roi_voxel.get_end()[1]),
                    range(roi_voxel.get_begin()[2], roi_voxel.get_end()[2]), indexing='ij')
            data = np.array(meshgrids)

            # print("Roi is: " + str(roi))

            spec = self.spec[array_key].copy()
            spec.roi = roi
            batch.arrays[array_key] = Array(
                    data,
                    spec)
        return batch

@skipIf(isinstance(zarr, NoSuchModule), 'zarr is not installed')
class TestZarrWrite(ProviderTest):

    def test_output(self):
        path = self.path_to('zarr_write_test.zarr')

        source = ZarrWriteTestSource()

        chunk_request = BatchRequest()
        chunk_request.add(ArrayKeys.RAW, (400,30,34))
        chunk_request.add(ArrayKeys.GT_LABELS, (200,10,14))

        pipeline = (
            source +
            ZarrWrite({
                ArrayKeys.RAW: 'arrays/raw'
            },
            output_filename=path) +
            Scan(chunk_request))

        with build(pipeline):

            raw_spec    = pipeline.spec[ArrayKeys.RAW]
            labels_spec = pipeline.spec[ArrayKeys.GT_LABELS]

            full_request = BatchRequest({
                    ArrayKeys.RAW: raw_spec,
                    ArrayKeys.GT_LABELS: labels_spec
                }
            )

            batch = pipeline.request_batch(full_request)

        # assert that stored HDF dataset equals batch array

        with ZarrFile(path, mode='r') as f:

            ds = f['arrays/raw']

            batch_raw = batch.arrays[ArrayKeys.RAW]
            stored_raw = ds[:]

            self.assertEqual(
                stored_raw.shape[-3:],
                batch_raw.spec.roi.shape//batch_raw.spec.voxel_size)
            self.assertEqual(tuple(ds.attrs['offset']), batch_raw.spec.roi.offset)
            self.assertEqual(tuple(ds.attrs['resolution']), batch_raw.spec.voxel_size)
            self.assertTrue((stored_raw == batch.arrays[ArrayKeys.RAW].data).all())

