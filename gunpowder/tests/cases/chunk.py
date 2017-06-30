from .provider_test import ProviderTest
from gunpowder import *
import numpy as np

class ChunkTestSource(BatchProvider):

    def get_spec(self):

        spec = ProviderSpec()
        spec.volumes[VolumeType.RAW] = Roi((0,0,0), (100,100,100))
        spec.volumes[VolumeType.GT_LABELS] = Roi((10,10,10), (90,90,90))
        return spec

    def provide(self, request):

        print("ChunkTestSource: Got request " + str(request))

        batch = Batch()

        # have the pixels encode their position
        for (volume_type, roi) in request.volumes.items():

            # print("ChunkTestSource: Adding " + str(volume_type))

            # the z,y,x coordinates of the ROI
            meshgrids = np.meshgrid(
                    range(roi.get_begin()[0], roi.get_end()[0]),
                    range(roi.get_begin()[1], roi.get_end()[1]),
                    range(roi.get_begin()[2], roi.get_end()[2]), indexing='ij')
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            # print("Roi is: " + str(roi))

            batch.volumes[volume_type] = Volume(
                    data,
                    roi,
                    (1,1,1),
                    False)
        return batch

class TestChunk(ProviderTest):

    def test_output(self):

        chunk_request = BatchRequest()
        chunk_request.add_volume_request(VolumeType.RAW, (20,15,17))
        chunk_request.add_volume_request(VolumeType.GT_LABELS, (10,5,7))

        full_request = BatchRequest()
        full_request.add_volume_request(VolumeType.RAW, (100, 100, 100))
        full_request.add_volume_request(VolumeType.GT_LABELS, (80, 80, 80))

        pipeline = ChunkTestSource() + Chunk(full_request, chunk_request, cache_size=20, num_workers=15)

        with build(pipeline):
            batch = pipeline.request_batch(full_request)

        # assert that pixels encode their position
        for (volume_type, volume) in batch.volumes.items():

            # the z,y,x coordinates of the ROI
            meshgrids = np.meshgrid(
                    range(volume.roi.get_begin()[0], volume.roi.get_end()[0]),
                    range(volume.roi.get_begin()[1], volume.roi.get_end()[1]),
                    range(volume.roi.get_begin()[2], volume.roi.get_end()[2]), indexing='ij')
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            self.assertTrue((volume.data == data).all())