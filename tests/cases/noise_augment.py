from .provider_test import ProviderTest
from gunpowder import IntensityAugment, ArrayKeys, build, Normalize, NoiseAugment


class TestIntensityAugment(ProviderTest):
    def test_shift(self):
        pipeline = (
            self.test_source
            + Normalize(ArrayKeys.RAW)
            + IntensityAugment(
                ArrayKeys.RAW, scale_min=0, scale_max=0, shift_min=0.5, shift_max=0.5
            )
            + NoiseAugment(ArrayKeys.RAW, clip=True)
        )

        with build(pipeline):
            for i in range(100):
                batch = pipeline.request_batch(self.test_request)

                x = batch.arrays[ArrayKeys.RAW].data
                assert x.min() < 0.5
                assert x.min() >= 0
                assert x.max() > 0.5
                assert x.max() <= 1
