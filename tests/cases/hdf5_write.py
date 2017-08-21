from .provider_test import ProviderTest
from gunpowder import *
import numpy as np
from gunpowder.ext import h5py

class Hdf5WriteTestSource(BatchProvider):

    def get_spec(self):
        spec = ProviderSpec()
        spec.volumes[VolumeTypes.RAW] = Roi((20000,2000,2000), (2000,200,200))
        spec.volumes[VolumeTypes.GT_LABELS] = Roi((20100,2010,2010), (1800,180,180))

        VolumeTypes.RAW.voxel_size       = (20,2,2)
        VolumeTypes.GT_LABELS.voxel_size = (20,2,2)

        return spec

    def provide(self, request):

        # print("Hdf5WriteTestSource: Got request " + str(request))

        batch = Batch()

        # have the pixels encode their position
        for (volume_type, roi) in request.volumes.items():

            roi_voxel = roi // volume_type.voxel_size
            # print("Hdf5WriteTestSource: Adding " + str(volume_type))

            # the z,y,x coordinates of the ROI
            meshgrids = np.meshgrid(
                    range(roi_voxel.get_begin()[0], roi_voxel.get_end()[0]),
                    range(roi_voxel.get_begin()[1], roi_voxel.get_end()[1]),
                    range(roi_voxel.get_begin()[2], roi_voxel.get_end()[2]), indexing='ij')
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            # print("Roi is: " + str(roi))

            batch.volumes[volume_type] = Volume(
                    data,
                    roi)
        return batch

class TestHdf5Write(ProviderTest):

    def test_output(self):

        source = Hdf5WriteTestSource()

        raw_roi    = source.get_spec().volumes[VolumeTypes.RAW]
        labels_roi = source.get_spec().volumes[VolumeTypes.GT_LABELS]

        chunk_request = BatchRequest()
        chunk_request.add_volume_request(VolumeTypes.RAW, (400,30,34))
        chunk_request.add_volume_request(VolumeTypes.GT_LABELS, (200,10,14))

        full_request = BatchRequest({
                VolumeTypes.RAW: raw_roi,
                VolumeTypes.GT_LABELS: labels_roi
            }
        )

        pipeline = source + Hdf5Write({VolumeTypes.RAW: 'volumes/raw'}, output_filename='hdf5_write_test.hdf') + Chunk(chunk_request)

        with build(pipeline):
            batch = pipeline.request_batch(full_request)

        # assert that stored HDF dataset equals batch volume

        with h5py.File('hdf5_write_test.hdf', 'r') as f:

            ds = f['volumes/raw']

            batch_raw = batch.volumes[VolumeTypes.RAW]
            stored_raw = np.array(ds)

            self.assertEqual(stored_raw.shape, batch_raw.roi.get_shape()//VolumeTypes.RAW.voxel_size)
            self.assertEqual(tuple(ds.attrs['offset']), batch_raw.roi.get_offset())
            self.assertEqual(tuple(ds.attrs['resolution']), VolumeTypes.RAW.voxel_size)

            print(stored_raw)
            print(batch.volumes[VolumeTypes.RAW].data)
            self.assertTrue((stored_raw == batch.volumes[VolumeTypes.RAW].data).all())
