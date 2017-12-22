from .provider_test import ProviderTest
from gunpowder import *
import logging
import numpy as np

class DownSampleTestSource(BatchProvider):

    def setup(self):

        self.provides(
            ArrayKeys.RAW,
            ArraySpec(
                roi=Roi((0, 0, 0), (1000, 1000, 1000)),
                voxel_size=(4, 4, 4)))

        self.provides(
            ArrayKeys.GT_LABELS,
            ArraySpec(
                roi=Roi((0, 0, 0), (1000, 1000, 1000)),
                voxel_size=(4, 4, 4)))

    def provide(self, request):

        batch = Batch()

        # have the pixels encode their position
        for (array_key, spec) in request.array_specs.items():

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

            spec = self.spec[array_key].copy()
            spec.roi = roi
            batch.arrays[array_key] = Array(
                    data,
                    spec)
        return batch

class TestDownSample(ProviderTest):

    def test_output(self):

        set_verbose(False)
        logger = logging.getLogger('gunpowder.nodes.downsample')
        logger.setLevel(logging.DEBUG)

        source = DownSampleTestSource()

        ArrayKey('RAW_DOWNSAMPLED')
        ArrayKey('GT_LABELS_DOWNSAMPLED')

        request = BatchRequest()
        request.add(ArrayKeys.RAW, (200,200,200))
        request.add(ArrayKeys.RAW_DOWNSAMPLED, (120,120,120))
        request.add(ArrayKeys.GT_LABELS, (200,200,200))
        request.add(ArrayKeys.GT_LABELS_DOWNSAMPLED, (200,200,200))

        pipeline = (
                DownSampleTestSource() +
                DownSample({
                        ArrayKeys.RAW_DOWNSAMPLED: (2, ArrayKeys.RAW),
                        ArrayKeys.GT_LABELS_DOWNSAMPLED: (2, ArrayKeys.GT_LABELS),
                })
        )

        with build(pipeline):
            batch = pipeline.request_batch(request)

        for (array_key, array) in batch.arrays.items():

            # assert that pixels encode their position for supposedly unaltered 
            # arrays
            if array_key in [ArrayKeys.RAW, ArrayKeys.GT_LABELS]:

                # the z,y,x coordinates of the ROI
                roi = array.spec.roi/4
                meshgrids = np.meshgrid(
                        range(roi.get_begin()[0], roi.get_end()[0]),
                        range(roi.get_begin()[1], roi.get_end()[1]),
                        range(roi.get_begin()[2], roi.get_end()[2]), indexing='ij')
                data = meshgrids[0] + meshgrids[1] + meshgrids[2]

                self.assertTrue(np.array_equal(array.data, data), str(array_key))

            elif array_key == ArrayKeys.RAW_DOWNSAMPLED:

                self.assertTrue(array.data[0,0,0] == 30)
                self.assertTrue(array.data[1,0,0] == 32)

            elif array_key == ArrayKeys.GT_LABELS_DOWNSAMPLED:

                self.assertTrue(array.data[0,0,0] == 0)
                self.assertTrue(array.data[1,0,0] == 2)

            else:

                self.assertTrue(False, "unexpected array type")
