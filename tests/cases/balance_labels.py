from .provider_test import ProviderTest
from gunpowder import *
import numpy as np

class TestSource(BatchProvider):

    def get_spec(self):

        spec = ProviderSpec()
        spec.volumes[VolumeTypes.GT_AFFINITIES] = Roi((0,0,0), (2000,200,200))
        spec.volumes[VolumeTypes.GT_MASK] = Roi((0,0,0), (2000,200,200))
        spec.volumes[VolumeTypes.GT_IGNORE] = Roi((0,0,0), (2000,200,200))

        return spec

    def provide(self, request):

        batch = Batch()

        roi = request.volumes[VolumeTypes.GT_AFFINITIES]
        shape_vx = request.volumes[VolumeTypes.GT_AFFINITIES].get_shape() // VolumeTypes.GT_AFFINITIES.voxel_size

        batch.volumes[VolumeTypes.GT_AFFINITIES] = Volume(
                np.random.randint(
                    0, 2,
                    (3,) + shape_vx
                ),
                roi
        )
        batch.volumes[VolumeTypes.GT_MASK] = Volume(
                np.random.randint(
                    0, 2,
                    shape_vx
                ),
                roi
        )
        batch.volumes[VolumeTypes.GT_IGNORE] = Volume(
                np.random.randint(
                    0, 2,
                    shape_vx
                ),
                roi
        )

        return batch

class TestBalanceLabels(ProviderTest):

    def test_output(self):

        voxel_size = (20, 2, 2)
        register_volume_type(VolumeType('GT_MASK', interpolate=False, voxel_size=voxel_size))
        register_volume_type(VolumeType('GT_IGNORE', interpolate=False, voxel_size=voxel_size))
        register_volume_type(VolumeType('LOSS_SCALE', interpolate=False, voxel_size=voxel_size))
        register_volume_type(VolumeType('GT_LABELS', interpolate=False, voxel_size=voxel_size))
        register_volume_type(VolumeType('GT_AFFINITIES', interpolate=False, voxel_size=voxel_size))

        pipeline = TestSource() + BalanceLabels({VolumeTypes.GT_AFFINITIES: VolumeTypes.LOSS_SCALE},
                                                {VolumeTypes.GT_AFFINITIES: [VolumeTypes.GT_MASK, VolumeTypes.GT_IGNORE]})

        with build(pipeline):

            # check correct scaling on 10 random samples
            for i in range(10):

                request = BatchRequest()
                request.add_volume_request(VolumeTypes.GT_AFFINITIES, (400,30,34))
                request.add_volume_request(VolumeTypes.LOSS_SCALE, (400,30,34))

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

        # restore default volume types
        voxel_size = (1,1,1)
        register_volume_type(VolumeType('GT_MASK', interpolate=False, voxel_size=voxel_size))
        register_volume_type(VolumeType('GT_IGNORE', interpolate=False, voxel_size=voxel_size))
        register_volume_type(VolumeType('LOSS_SCALE', interpolate=False, voxel_size=voxel_size))
        register_volume_type(VolumeType('GT_LABELS', interpolate=False, voxel_size=voxel_size))
        register_volume_type(VolumeType('GT_AFFINTIIES', interpolate=False, voxel_size=voxel_size))