from .provider_test import ProviderTest
from gunpowder import *
import numpy as np
from gunpowder.ext import h5py

class Hdf5WriteTestSource(BatchProvider):

    def setup(self):

        self.provides(
            VolumeTypes.RAW,
            VolumeSpec(
                roi=Roi((20000, 2000, 2000), (2000, 200, 200)),
                voxel_size=(20, 2, 2)))
        self.provides(
            VolumeTypes.GT_LABELS,
            VolumeSpec(
                roi=Roi((20100, 2010, 2010), (1800, 180, 180)),
                voxel_size=(20, 2, 2)))

    def provide(self, request):

        # print("Hdf5WriteTestSource: Got request " + str(request))

        batch = Batch()

        # have the pixels encode their position
        for (volume_type, spec) in request.volume_specs.items():

            roi = spec.roi
            roi_voxel = roi // self.spec[volume_type].voxel_size
            # print("Hdf5WriteTestSource: Adding " + str(volume_type))

            # the z,y,x coordinates of the ROI
            meshgrids = np.meshgrid(
                    range(roi_voxel.get_begin()[0], roi_voxel.get_end()[0]),
                    range(roi_voxel.get_begin()[1], roi_voxel.get_end()[1]),
                    range(roi_voxel.get_begin()[2], roi_voxel.get_end()[2]), indexing='ij')
            data = np.array(meshgrids)
            print(data.shape)

            # print("Roi is: " + str(roi))

            spec = self.spec[volume_type].copy()
            spec.roi = roi
            batch.volumes[volume_type] = Volume(
                    data,
                    spec)
        return batch

class TestHdf5Write(ProviderTest):

    def test_output(self):

        source = Hdf5WriteTestSource()

        chunk_request = BatchRequest()
        chunk_request.add(VolumeTypes.RAW, (400,30,34))
        chunk_request.add(VolumeTypes.GT_LABELS, (200,10,14))

        pipeline = (
            source +
            Hdf5Write({
                VolumeTypes.RAW: 'volumes/raw'
            },
            output_filename='hdf5_write_test.hdf') +
            Scan(chunk_request))

        with build(pipeline):

            raw_spec    = pipeline.spec[VolumeTypes.RAW]
            labels_spec = pipeline.spec[VolumeTypes.GT_LABELS]

            full_request = BatchRequest({
                    VolumeTypes.RAW: raw_spec,
                    VolumeTypes.GT_LABELS: labels_spec
                }
            )

            batch = pipeline.request_batch(full_request)

        # assert that stored HDF dataset equals batch volume

        with h5py.File('hdf5_write_test.hdf', 'r') as f:

            ds = f['volumes/raw']

            batch_raw = batch.volumes[VolumeTypes.RAW]
            stored_raw = np.array(ds)

            self.assertEqual(
                stored_raw.shape[-3:],
                batch_raw.spec.roi.get_shape()//batch_raw.spec.voxel_size)
            self.assertEqual(tuple(ds.attrs['offset']), batch_raw.spec.roi.get_offset())
            self.assertEqual(tuple(ds.attrs['resolution']), batch_raw.spec.voxel_size)
            self.assertTrue((stored_raw == batch.volumes[VolumeTypes.RAW].data).all())
