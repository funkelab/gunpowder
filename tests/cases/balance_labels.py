from .provider_test import ProviderTest
from gunpowder import *
import numpy as np


class ExampleSource(BatchProvider):
    def setup(self):
        for identifier in [
            ArrayKeys.GT_AFFINITIES,
            ArrayKeys.GT_AFFINITIES_MASK,
            ArrayKeys.GT_IGNORE,
        ]:
            self.provides(
                identifier,
                ArraySpec(roi=Roi((0, 0, 0), (2000, 200, 200)), voxel_size=(20, 2, 2)),
            )

    def provide(self, request):
        batch = Batch()

        roi = request[ArrayKeys.GT_AFFINITIES].roi
        shape_vx = roi.shape // self.spec[ArrayKeys.GT_AFFINITIES].voxel_size

        spec = self.spec[ArrayKeys.GT_AFFINITIES].copy()
        spec.roi = roi

        batch.arrays[ArrayKeys.GT_AFFINITIES] = Array(
            np.random.randint(0, 2, (3,) + shape_vx), spec
        )
        batch.arrays[ArrayKeys.GT_AFFINITIES_MASK] = Array(
            np.random.randint(0, 2, (3,) + shape_vx), spec
        )
        batch.arrays[ArrayKeys.GT_IGNORE] = Array(
            np.random.randint(0, 2, (3,) + shape_vx), spec
        )

        return batch


class TestBalanceLabels(ProviderTest):
    def test_output(self):
        pipeline = ExampleSource() + BalanceLabels(
            labels=ArrayKeys.GT_AFFINITIES,
            scales=ArrayKeys.LOSS_SCALE,
            mask=[ArrayKeys.GT_AFFINITIES_MASK, ArrayKeys.GT_IGNORE],
        )

        with build(pipeline):
            # check correct scaling on 10 random samples
            for i in range(10):
                request = BatchRequest()
                request.add(ArrayKeys.GT_AFFINITIES, (400, 30, 34))
                request.add(ArrayKeys.GT_AFFINITIES_MASK, (400, 30, 34))
                request.add(ArrayKeys.GT_IGNORE, (400, 30, 34))
                request.add(ArrayKeys.LOSS_SCALE, (400, 30, 34))

                batch = pipeline.request_batch(request)

                self.assertTrue(ArrayKeys.LOSS_SCALE in batch.arrays)

                affs = batch.arrays[ArrayKeys.GT_AFFINITIES].data
                scale = batch.arrays[ArrayKeys.LOSS_SCALE].data
                mask = batch.arrays[ArrayKeys.GT_AFFINITIES_MASK].data
                ignore = batch.arrays[ArrayKeys.GT_IGNORE].data

                # combine mask and ignore
                mask *= ignore

                self.assertTrue((scale[mask == 1] > 0).all())
                self.assertTrue((scale[mask == 0] == 0).all())

                num_masked_out = affs.size - mask.sum()
                num_masked_in = affs.size - num_masked_out
                num_pos = (affs * mask).sum()
                num_neg = affs.size - num_masked_out - num_pos

                frac_pos = float(num_pos) / num_masked_in if num_masked_in > 0 else 0
                frac_pos = min(0.95, max(0.05, frac_pos))
                frac_neg = 1.0 - frac_pos

                w_pos = 1.0 / (2.0 * frac_pos)
                w_neg = 1.0 / (2.0 * frac_neg)

                self.assertAlmostEqual((scale * mask * affs).sum(), w_pos * num_pos, 3)
                self.assertAlmostEqual(
                    (scale * mask * (1 - affs)).sum(), w_neg * num_neg, 3
                )

                # check if LOSS_SCALE is omitted if not requested
                del request[ArrayKeys.LOSS_SCALE]

                batch = pipeline.request_batch(request)
                self.assertTrue(ArrayKeys.LOSS_SCALE not in batch.arrays)

        # same using a slab for balancing

        pipeline = ExampleSource() + BalanceLabels(
            labels=ArrayKeys.GT_AFFINITIES,
            scales=ArrayKeys.LOSS_SCALE,
            mask=[ArrayKeys.GT_AFFINITIES_MASK, ArrayKeys.GT_IGNORE],
            slab=(1, -1, -1, -1),
        )  # every channel individually

        with build(pipeline):
            # check correct scaling on 10 random samples
            for i in range(10):
                request = BatchRequest()
                request.add(ArrayKeys.GT_AFFINITIES, (400, 30, 34))
                request.add(ArrayKeys.GT_AFFINITIES_MASK, (400, 30, 34))
                request.add(ArrayKeys.GT_IGNORE, (400, 30, 34))
                request.add(ArrayKeys.LOSS_SCALE, (400, 30, 34))

                batch = pipeline.request_batch(request)

                self.assertTrue(ArrayKeys.LOSS_SCALE in batch.arrays)

                for c in range(3):
                    affs = batch.arrays[ArrayKeys.GT_AFFINITIES].data[c]
                    scale = batch.arrays[ArrayKeys.LOSS_SCALE].data[c]
                    mask = batch.arrays[ArrayKeys.GT_AFFINITIES_MASK].data[c]
                    ignore = batch.arrays[ArrayKeys.GT_IGNORE].data[c]

                    # combine mask and ignore
                    mask *= ignore

                    self.assertTrue((scale[mask == 1] > 0).all())
                    self.assertTrue((scale[mask == 0] == 0).all())

                    num_masked_out = affs.size - mask.sum()
                    num_masked_in = affs.size - num_masked_out
                    num_pos = (affs * mask).sum()
                    num_neg = affs.size - num_masked_out - num_pos

                    frac_pos = (
                        float(num_pos) / num_masked_in if num_masked_in > 0 else 0
                    )
                    frac_pos = min(0.95, max(0.05, frac_pos))
                    frac_neg = 1.0 - frac_pos

                    w_pos = 1.0 / (2.0 * frac_pos)
                    w_neg = 1.0 / (2.0 * frac_neg)

                    self.assertAlmostEqual(
                        (scale * mask * affs).sum(), w_pos * num_pos, 3
                    )
                    self.assertAlmostEqual(
                        (scale * mask * (1 - affs)).sum(), w_neg * num_neg, 3
                    )
