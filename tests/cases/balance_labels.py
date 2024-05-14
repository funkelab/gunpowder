import numpy as np

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    BalanceLabels,
    BatchRequest,
    MergeProvider,
    Roi,
    build,
)

from .helper_sources import ArraySource


def test_output():
    affs_key = ArrayKey("AFFS")
    affs_mask_key = ArrayKey("AFFS_MASK")
    ignore_key = ArrayKey("IGNORE")
    loss_scale_key = ArrayKey("LOSS_SCALE")

    array_spec = ArraySpec(roi=Roi((0, 0, 0), (2000, 200, 200)), voxel_size=(20, 2, 2))

    data_shape = array_spec.roi.shape // array_spec.voxel_size
    affs_data = np.random.randint(0, 2, (3,) + data_shape)
    affs_mask_data = np.random.randint(0, 2, (3,) + data_shape)
    ignore_data = np.random.randint(0, 2, (3,) + data_shape)

    affs_array = Array(affs_data, array_spec.copy())
    affs_mask_array = Array(affs_mask_data, array_spec.copy())
    ignore_array = Array(ignore_data, array_spec.copy())

    pipeline = (
        (
            ArraySource(affs_key, affs_array),
            ArraySource(affs_mask_key, affs_mask_array),
            ArraySource(ignore_key, ignore_array),
        )
        + MergeProvider()
        + BalanceLabels(
            labels=affs_key,
            scales=loss_scale_key,
            mask=[affs_mask_key, ignore_key],
        )
    )

    with build(pipeline):
        # check correct scaling on 10 random samples
        for i in range(10):
            request = BatchRequest()
            request.add(affs_key, (400, 30, 34))
            request.add(affs_mask_key, (400, 30, 34))
            request.add(ignore_key, (400, 30, 34))
            request.add(loss_scale_key, (400, 30, 34))

            batch = pipeline.request_batch(request)

            assert loss_scale_key in batch.arrays

            affs = batch.arrays[affs_key].data
            scale = batch.arrays[loss_scale_key].data
            mask = batch.arrays[affs_mask_key].data
            ignore = batch.arrays[ignore_key].data

            # combine mask and ignore
            mask *= ignore

            assert (scale[mask == 1] > 0).all()
            assert (scale[mask == 0] == 0).all()

            num_masked_out = affs.size - mask.sum()
            num_masked_in = affs.size - num_masked_out
            num_pos = (affs * mask).sum()
            num_neg = affs.size - num_masked_out - num_pos

            frac_pos = float(num_pos) / num_masked_in if num_masked_in > 0 else 0
            frac_pos = min(0.95, max(0.05, frac_pos))
            frac_neg = 1.0 - frac_pos

            w_pos = 1.0 / (2.0 * frac_pos)
            w_neg = 1.0 / (2.0 * frac_neg)

            assert abs((scale * mask * affs).sum() - w_pos * num_pos) < 1e-3
            assert abs((scale * mask * (1 - affs)).sum() - w_neg * num_neg < 1e-3)

            # check if LOSS_SCALE is omitted if not requested
            del request[loss_scale_key]

            batch = pipeline.request_batch(request)
            assert loss_scale_key not in batch.arrays

    # same using a slab for balancing

    pipeline = (
        (
            ArraySource(affs_key, affs_array),
            ArraySource(affs_mask_key, affs_mask_array),
            ArraySource(ignore_key, ignore_array),
        )
        + MergeProvider()
        + BalanceLabels(
            labels=affs_key,
            scales=loss_scale_key,
            mask=[affs_mask_key, ignore_key],
            slab=(1, -1, -1, -1),  # every channel individually
        )
    )

    with build(pipeline):
        # check correct scaling on 10 random samples
        for i in range(10):
            request = BatchRequest()
            request.add(affs_key, (400, 30, 34))
            request.add(affs_mask_key, (400, 30, 34))
            request.add(ignore_key, (400, 30, 34))
            request.add(loss_scale_key, (400, 30, 34))

            batch = pipeline.request_batch(request)

            assert loss_scale_key in batch.arrays

            for c in range(3):
                affs = batch.arrays[affs_key].data[c]
                scale = batch.arrays[loss_scale_key].data[c]
                mask = batch.arrays[affs_mask_key].data[c]
                ignore = batch.arrays[ignore_key].data[c]

                # combine mask and ignore
                mask *= ignore

                assert (scale[mask == 1] > 0).all()
                assert (scale[mask == 0] == 0).all()

                num_masked_out = affs.size - mask.sum()
                num_masked_in = affs.size - num_masked_out
                num_pos = (affs * mask).sum()
                num_neg = affs.size - num_masked_out - num_pos

                frac_pos = float(num_pos) / num_masked_in if num_masked_in > 0 else 0
                frac_pos = min(0.95, max(0.05, frac_pos))
                frac_neg = 1.0 - frac_pos

                w_pos = 1.0 / (2.0 * frac_pos)
                w_neg = 1.0 / (2.0 * frac_neg)

                assert abs((scale * mask * affs).sum() - w_pos * num_pos) < 1e-3
                assert abs((scale * mask * (1 - affs)).sum() - w_neg * num_neg) < 1e-3
