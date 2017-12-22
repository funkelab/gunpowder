from .provider_test import ProviderTest
from gunpowder import *
import logging
import numpy as np

class DownSampleTestSource(BatchProvider):

    def setup(self):

        self.provides(
            ArrayTypes.RAW,
            ArraySpec(
                roi=Roi((0, 0, 0), (1000, 1000, 1000)),
                voxel_size=(4, 4, 4)))

        self.provides(
            ArrayTypes.GT_LABELS,
            ArraySpec(
                roi=Roi((0, 0, 0), (1000, 1000, 1000)),
                voxel_size=(4, 4, 4)))

    def provide(self, request):

        batch = Batch()

        # have the pixels encode their position
        for (volume_type, spec) in request.volume_specs.items():

            roi = spec.roi

            for d in range(3):
                assert roi.get_begin()[d]%4 == 0, "roi %s does not align with voxels"

            data_roi = roi/4

            # the z,y,x coordinates of the ROI
            meshgrids = np.meshgrid(
                    range(data_roi.get_begin()[0], data_roi.get_end()[0]),
                    range(data_roi.get_begin()[1], data_roi.get_end()[1]),
                    range(data_roi.get_begin()[2], data_roi.get_end()[2]), indexing='ij')
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            spec = self.spec[volume_type].copy()
            spec.roi = roi
            batch.volumes[volume_type] = Array(
                    data,
                    spec)
        return batch

class TestDownSample(ProviderTest):

    def test_output(self):

        set_verbose(False)
        logger = logging.getLogger('gunpowder.nodes.downsample')
        logger.setLevel(logging.DEBUG)

        source = DownSampleTestSource()

        register_volume_type('RAW_DOWNSAMPLED')
        register_volume_type('GT_LABELS_DOWNSAMPLED')

        request = BatchRequest()
        request.add(ArrayTypes.RAW, (200,200,200))
        request.add(ArrayTypes.RAW_DOWNSAMPLED, (120,120,120))
        request.add(ArrayTypes.GT_LABELS, (200,200,200))
        request.add(ArrayTypes.GT_LABELS_DOWNSAMPLED, (200,200,200))

        pipeline = (
                DownSampleTestSource() +
                DownSample({
                        ArrayTypes.RAW_DOWNSAMPLED: (2, ArrayTypes.RAW),
                        ArrayTypes.GT_LABELS_DOWNSAMPLED: (2, ArrayTypes.GT_LABELS),
                })
        )

        with build(pipeline):
            batch = pipeline.request_batch(request)

        for (volume_type, volume) in batch.volumes.items():

            # assert that pixels encode their position for supposedly unaltered 
            # volumes
            if volume_type in [ArrayTypes.RAW, ArrayTypes.GT_LABELS]:

                # the z,y,x coordinates of the ROI
                roi = volume.spec.roi/4
                meshgrids = np.meshgrid(
                        range(roi.get_begin()[0], roi.get_end()[0]),
                        range(roi.get_begin()[1], roi.get_end()[1]),
                        range(roi.get_begin()[2], roi.get_end()[2]), indexing='ij')
                data = meshgrids[0] + meshgrids[1] + meshgrids[2]

                self.assertTrue(np.array_equal(volume.data, data), str(volume_type))

            elif volume_type == ArrayTypes.RAW_DOWNSAMPLED:

                self.assertTrue(volume.data[0,0,0] == 30)
                self.assertTrue(volume.data[1,0,0] == 32)

            elif volume_type == ArrayTypes.GT_LABELS_DOWNSAMPLED:

                self.assertTrue(volume.data[0,0,0] == 0)
                self.assertTrue(volume.data[1,0,0] == 2)

            else:

                self.assertTrue(False, "unexpected volume type")
