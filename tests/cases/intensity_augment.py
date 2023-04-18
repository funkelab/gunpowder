from .provider_test import ProviderTest
from gunpowder import IntensityAugment, ArrayKeys, build, Normalize

import numpy as np


class TestIntensityAugment(ProviderTest):
    def test_shift(self):
        pipeline = (
            self.test_source
            + Normalize(ArrayKeys.RAW)
            + IntensityAugment(
                ArrayKeys.RAW, scale_min=0, scale_max=0, shift_min=0.5, shift_max=0.5
            )
        )

        with build(pipeline):
            for i in range(100):
                batch = pipeline.request_batch(self.test_request)

                x = batch.arrays[ArrayKeys.RAW].data
                assert np.isclose(x.min(), 0.5)
                assert np.isclose(x.max(), 0.5)
