from .provider_test import ProviderTest
from gunpowder import *
import numpy as np

class TestSource(BatchProvider):

    def get_spec(self):

        spec = ProviderSpec()
        spec.volumes[VolumeTypes.GT_AFFINITIES] = Roi((0,0,0), (100,100,100))
        spec.volumes[VolumeTypes.GT_MASK] = Roi((0,0,0), (100,100,100))
        spec.volumes[VolumeTypes.GT_IGNORE] = Roi((0,0,0), (100,100,100))
        return spec

    def provide(self, request):

        batch = Batch()

        roi = request.volumes[VolumeTypes.GT_AFFINITIES]
        shape = request.volumes[VolumeTypes.GT_AFFINITIES].get_shape()

        batch.volumes[VolumeTypes.GT_AFFINITIES] = Volume(
                np.random.randint(
                    0, 2,
                    (3,) + shape
                ),
                roi,
                (1,1,1)
        )
        batch.volumes[VolumeTypes.GT_MASK] = Volume(
                np.random.randint(
                    0, 2,
                    shape
                ),
                roi,
                (1,1,1)
        )
        batch.volumes[VolumeTypes.GT_IGNORE] = Volume(
                np.random.randint(
                    0, 2,
                    shape
                ),
                roi,
                (1,1,1)
        )

        return batch

class TestBalanceLabels(ProviderTest):

    def test_output(self):

        pipeline = TestSource() + BalanceLabels()

        with build(pipeline):

            # check correct scaling on 10 random samples
            for i in range(10):

                request = BatchRequest()
                request.add_volume_request(VolumeTypes.GT_AFFINITIES, (20,15,17))
                request.add_volume_request(VolumeTypes.LOSS_SCALE, (20,15,17))

                batch = pipeline.request_batch(request)

                self.assertTrue(VolumeTypes.LOSS_SCALE in batch.volumes)

                affs = batch.volumes[VolumeTypes.GT_AFFINITIES].data
                scale = batch.volumes[VolumeTypes.LOSS_SCALE].data
                mask = batch.volumes[VolumeTypes.GT_MASK].data
                ignore = batch.volumes[VolumeTypes.GT_IGNORE].data

                # combine mask and ignore
                mask *= ignore

                # make a mask on affinities
                mask = np.array([mask, mask, mask])

                self.assertTrue((scale[mask==1] > 0).all())
                self.assertTrue((scale[mask==0] == 0).all())

                num_masked_out = affs.size - mask.sum()
                num_masked_in = affs.size - num_masked_out
                num_pos = (affs*mask).sum()
                num_neg = affs.size - num_masked_out - num_pos

                frac_pos = float(num_pos)/num_masked_in if num_masked_in > 0 else 0
                frac_pos = min(0.95, max(0.05, frac_pos))
                frac_neg = 1.0 - frac_pos

                w_pos = 1.0/(2.0*frac_pos)
                w_neg = 1.0/(2.0*frac_neg)

                self.assertAlmostEqual((scale*mask*affs).sum(), w_pos*num_pos)
                self.assertAlmostEqual((scale*mask*(1-affs)).sum(), w_neg*num_neg)

                # check if LOSS_SCALE is omitted if not requested
                del request.volumes[VolumeTypes.LOSS_SCALE]

                batch = pipeline.request_batch(request)
                self.assertTrue(VolumeTypes.LOSS_SCALE not in batch.volumes)
