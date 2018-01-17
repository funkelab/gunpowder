from .provider_test import ProviderTest
from gunpowder import *
import numpy as np

class TestDvidSource(ProviderTest):

    def test_output_3d(self):

        # create array keys
        raw = ArrayKey('RAW')
        seg = ArrayKey('SEG')
        mask = ArrayKey('MASK')

        source = DvidSource(
            'slowpoke1',
            32768,
            '2ad1d8f0f172425c9f87b60fd97331e6',
            datasets = {
                raw: 'grayscale',
                seg: 'groundtruth'
            },
            masks = {
                mask: 'seven_column'
            }
        )

        with build(source):

            batch = source.request_batch(
                BatchRequest({
                    raw: ArraySpec(roi=Roi((0, 0, 0), (80, 80, 80))),
                    seg: ArraySpec(roi=Roi((0, 0, 0), (80, 80, 80))),
                    mask: ArraySpec(roi=Roi((0, 0, 0), (80, 80, 80)))
                })
            )

            self.assertTrue(batch.arrays[raw].spec.interpolatable)
            self.assertFalse(batch.arrays[seg].spec.interpolatable)
            self.assertFalse(batch.arrays[mask].spec.interpolatable)

