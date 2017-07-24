from .provider_test import ProviderTest
from gunpowder import *
import logging
import numpy as np

class DownSampleTestSource(BatchProvider):

    def get_spec(self):

        spec = ProviderSpec()
        spec.volumes[VolumeTypes.RAW] = Roi((1000,1000,1000), (100,100,100))
        spec.volumes[VolumeTypes.GT_LABELS] = Roi((1005,1005,1005), (90,90,90))
        return spec

    def provide(self, request):

        batch = Batch()

        # have the pixels encode their position
        for (volume_type, roi) in request.volumes.items():

            # the z,y,x coordinates of the ROI
            meshgrids = np.meshgrid(
                    range(roi.get_begin()[0], roi.get_end()[0]),
                    range(roi.get_begin()[1], roi.get_end()[1]),
                    range(roi.get_begin()[2], roi.get_end()[2]), indexing='ij')
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            batch.volumes[volume_type] = Volume(
                    data,
                    roi,
                    Coordinate((4,4,4)))
        return batch

class TestDownSample(ProviderTest):

    def test_output(self):

        logger = logging.getLogger('gunpowder.nodes.downsample')
        logger.setLevel(logging.DEBUG)

        source = DownSampleTestSource()

        register_volume_type(VolumeType('RAW_DOWNSAMPLED', interpolate=True))
        register_volume_type(VolumeType('GT_LABELS_DOWNSAMPLED', interpolate=False))

        request = BatchRequest()
        request.add_volume_request(VolumeTypes.RAW, (100,100,100))
        request.add_volume_request(VolumeTypes.RAW_DOWNSAMPLED, (50,50,50))
        request.add_volume_request(VolumeTypes.GT_LABELS, (100,100,100))
        request.add_volume_request(VolumeTypes.GT_LABELS_DOWNSAMPLED, (60,60,60))

        pipeline = (
                DownSampleTestSource() +
                DownSample({
                        VolumeTypes.RAW: (2, VolumeTypes.RAW_DOWNSAMPLED),
                        VolumeTypes.GT_LABELS: (2, VolumeTypes.GT_LABELS_DOWNSAMPLED),
                })
        )

        with build(pipeline):
            batch = pipeline.request_batch(request)

        for (volume_type, volume) in batch.volumes.items():

            # assert that pixels encode their position for supposedly unaltered 
            # volumes
            if volume_type in [VolumeTypes.RAW, VolumeTypes.GT_LABELS]:

                # the z,y,x coordinates of the ROI
                meshgrids = np.meshgrid(
                        range(volume.roi.get_begin()[0], volume.roi.get_end()[0]),
                        range(volume.roi.get_begin()[1], volume.roi.get_end()[1]),
                        range(volume.roi.get_begin()[2], volume.roi.get_end()[2]), indexing='ij')
                data = meshgrids[0] + meshgrids[1] + meshgrids[2]

                self.assertTrue((volume.data == data).all(), str(volume_type))
                self.assertTrue(volume.resolution == (4,4,4))

            elif volume_type == VolumeTypes.RAW_DOWNSAMPLED:

                self.assertTrue(volume.data[0,0,0] == 0)
                self.assertTrue(volume.data[1,0,0] == 2)
                self.assertTrue(volume.resolution == (8,8,8))

            elif volume_type == VolumeTypes.GT_LABELS_DOWNSAMPLED:

                self.assertTrue(volume.data[0,0,0] == -30)
                self.assertTrue(volume.data[1,0,0] == -28)
                self.assertTrue(volume.resolution == (8,8,8))

            else:

                self.assertTrue(False, "unexpected volume type")
