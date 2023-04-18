from .provider_test import ProviderTest
from gunpowder import *
import numpy as np


class AsTypeTestSource(BatchProvider):
    def setup(self):
        self.provides(
            ArrayKeys.RAW,
            ArraySpec(roi=Roi((0, 0, 0), (1000, 1000, 1000)), voxel_size=(4, 4, 4)),
        )

        self.provides(
            ArrayKeys.GT_LABELS,
            ArraySpec(roi=Roi((0, 0, 0), (1000, 1000, 1000)), voxel_size=(4, 4, 4)),
        )

    def provide(self, request):
        batch = Batch()

        # have the pixels encode their position
        for array_key, spec in request.array_specs.items():
            roi = spec.roi

            data_roi = roi / 4

            # the z,y,x coordinates of the ROI
            meshgrids = np.meshgrid(
                range(data_roi.get_begin()[0], data_roi.get_end()[0]),
                range(data_roi.get_begin()[1], data_roi.get_end()[1]),
                range(data_roi.get_begin()[2], data_roi.get_end()[2]),
                indexing="ij",
            )
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            spec = self.spec[array_key].copy()
            spec.roi = roi
            batch.arrays[array_key] = Array(data, spec)
        return batch


class TestAsType(ProviderTest):
    def test_output(self):
        ArrayKey("RAW_TYPECAST")
        ArrayKey("GT_LABELS_TYPECAST")

        request = BatchRequest()
        request.add(ArrayKeys.RAW, (200, 200, 200))
        request.add(ArrayKeys.RAW_TYPECAST, (120, 120, 120))
        request.add(ArrayKeys.GT_LABELS, (200, 200, 200))
        request.add(ArrayKeys.GT_LABELS_TYPECAST, (200, 200, 200))

        pipeline = (
            AsTypeTestSource()
            + AsType(ArrayKeys.RAW, np.float16, ArrayKeys.RAW_TYPECAST)
            + AsType(ArrayKeys.GT_LABELS, np.int16, ArrayKeys.GT_LABELS_TYPECAST)
        )

        with build(pipeline):
            batch = pipeline.request_batch(request)

        for array_key, array in batch.arrays.items():
            # assert that pixels encode their position for supposedly unaltered
            # arrays
            if array_key in [ArrayKeys.RAW, ArrayKeys.GT_LABELS]:
                # the z,y,x coordinates of the ROI
                roi = array.spec.roi / 4
                meshgrids = np.meshgrid(
                    range(roi.get_begin()[0], roi.get_end()[0]),
                    range(roi.get_begin()[1], roi.get_end()[1]),
                    range(roi.get_begin()[2], roi.get_end()[2]),
                    indexing="ij",
                )
                data = meshgrids[0] + meshgrids[1] + meshgrids[2]

                self.assertTrue(np.array_equal(array.data, data), str(array_key))

            elif array_key == ArrayKeys.RAW_TYPECAST:
                self.assertTrue(
                    array.data.dtype == np.float16,
                    f"RAW_TYPECAST dtype: {array.data.dtype} does not equal expected: np.float16",
                )
                self.assertTrue(
                    int(array.data[1, 11, 1]) == 43,
                    f"RAW_TYPECAST[1,11,1]: int({array.data[1,11,1]}) does not equal expected: 43",
                )

            elif array_key == ArrayKeys.GT_LABELS_TYPECAST:
                self.assertTrue(
                    array.data.dtype == np.int16,
                    f"GT_LABELS_TYPECAST dtype: {array.data.dtype} does not equal expected: np.int16",
                )
                self.assertTrue(
                    int(array.data[1, 11, 1]) == 13,
                    f"GT_LABELS_TYPECAST[1,11,1]: int({array.data[1,11,1]}) does not equal expected: 13",
                )

            else:
                self.assertTrue(False, "unexpected array type")
