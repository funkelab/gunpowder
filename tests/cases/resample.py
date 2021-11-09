from .provider_test import ProviderTest
from gunpowder import *
import numpy as np

class ResampleTestSource(BatchProvider):

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

class TestResample(ProviderTest):

    def test_output(self):

        ArrayKey('RAW_RESAMPLED')
        ArrayKey('GT_LABELS_RESAMPLED')

        request = BatchRequest()
        request.add(ArrayKeys.RAW, (200,200,200))
        request.add(ArrayKeys.RAW_RESAMPLED, (120,120,120))
        request.add(ArrayKeys.GT_LABELS, (200,200,200))
        request.add(ArrayKeys.GT_LABELS_RESAMPLED, (200,200,200))

        pipeline = (
                ResampleTestSource() +
                Resample(ArrayKeys.RAW, Coordinate((8,8,8)), ArrayKeys.RAW_RESAMPLED, interp_order=0) +
                Resample(ArrayKeys.GT_LABELS, Coordinate((8,8,8)), ArrayKeys.GT_LABELS_RESAMPLED, interp_order=0)
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

            elif array_key == ArrayKeys.RAW_RESAMPLED:

                self.assertTrue(array.data[0,0,0] == 30, 
                                f'RAW_RESAMPLED[0,0,0]: {array.data[0,0,0]} does not equal expected: 30')
                self.assertTrue(array.data[1,0,0] == 32, 
                                f'RAW_RESAMPLED[1,0,0]: {array.data[1,0,0]} does not equal expected: 32')

            elif array_key == ArrayKeys.GT_LABELS_RESAMPLED:

                self.assertTrue(array.data[0,0,0] == 0, 
                                f'GT_LABELS_RESAMPLED[0,0,0]: {array.data[0,0,0]} does not equal expected: 0')
                self.assertTrue(array.data[1,0,0] == 2, 
                                f'GT_LABELS_RESAMPLED[1,0,0]: {array.data[1,0,0]} does not equal expected: 2')

            else:

                self.assertTrue(False, "unexpected array type")
