from .provider_test import ProviderTest
from gunpowder import *
import numpy as np

class TestSource(BatchProvider):

    def setup(self):

        self.provides(
            VolumeTypes.GT_LABELS, VolumeSpec(
                roi=Roi((-40, -40, -40), (160, 160, 160)),
                voxel_size=(20, 4, 8),
                interpolatable=False))

    def provide(self, request):

        batch = Batch()

        roi = request[VolumeTypes.GT_LABELS].roi
        shape = (roi/self.spec[VolumeTypes.GT_LABELS].voxel_size).get_shape()

        spec = self.spec[VolumeTypes.GT_LABELS].copy()
        spec.roi = roi
        data = np.ones(shape)
        data[shape[0]/2:,:,:] += 2
        data[:,shape[1]/2:,:] += 4
        data[:,:,shape[2]/2:] += 8
        batch.volumes[VolumeTypes.GT_LABELS] = Volume(data, spec)

        return batch

class TestAddBoundaryDistanceGradients(ProviderTest):

    def test_output(self):

        # set_verbose()

        register_volume_type('GT_BOUNDARY_DISTANCES')
        register_volume_type('GT_BOUNDARY_GRADIENTS')

        pipeline = (
            TestSource() +
            AddBoundaryDistanceGradients(
                distance_volume_type=VolumeTypes.GT_BOUNDARY_DISTANCES,
                gradient_volume_type=VolumeTypes.GT_BOUNDARY_GRADIENTS)
        )

        with build(pipeline):

            request = BatchRequest()
            request.add(VolumeTypes.GT_LABELS, (120,16,64))
            request.add(VolumeTypes.GT_BOUNDARY_DISTANCES, (120,16,64))
            request.add(VolumeTypes.GT_BOUNDARY_GRADIENTS, (120,16,64))

            batch = pipeline.request_batch(request)

            labels = batch.volumes[VolumeTypes.GT_LABELS].data
            distances = batch.volumes[VolumeTypes.GT_BOUNDARY_DISTANCES].data
            gradients = batch.volumes[VolumeTypes.GT_BOUNDARY_GRADIENTS].data
            shape = distances.shape

            l_001 = labels[:shape[0]/2,:shape[1]/2,shape[2]/2:]
            l_101 = labels[shape[0]/2:,:shape[1]/2,shape[2]/2:]
            d_001 = distances[:shape[0]/2,:shape[1]/2,shape[2]/2:]
            d_101 = distances[shape[0]/2:,:shape[1]/2,shape[2]/2:]
            g_001 = gradients[:,:shape[0]/2,:shape[1]/2,shape[2]/2:]
            g_101 = gradients[:,shape[0]/2:,:shape[1]/2,shape[2]/2:]

            # print labels
            # print
            # print distances
            # print
            # print l_001
            # print l_101
            # print
            # print d_001
            # print d_101
            # print
            # print g_001
            # print g_101

            self.assertTrue((g_001 == g_101).all())

            top = gradients[:,0:shape[0]/2,:]
            bot = gradients[:,shape[0]:shape[0]/2-1:-1,:]

            self.assertTrue((top == bot).all())
