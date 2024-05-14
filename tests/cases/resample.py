import numpy as np

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    BatchRequest,
    Coordinate,
    MergeProvider,
    Resample,
    Roi,
    build,
)

from .helper_sources import ArraySource


def test_up_and_downsample():
    meshgrids = np.meshgrid(range(0, 250), range(0, 250), range(0, 250), indexing="ij")
    array = meshgrids[0] + meshgrids[1] + meshgrids[2]
    array = np.stack([array, array, array], axis=0)

    raw_key = ArrayKey("RAW")
    raw_resampled_key = ArrayKey("RAW_RESAMPLED")
    gt_key = ArrayKey("GT")
    gt_resampled_key = ArrayKey("GT_LABELS_RESAMPLED")

    raw_source = ArraySource(
        raw_key,
        Array(
            array, ArraySpec(Roi((0, 0, 0), (1000, 1000, 1000)), Coordinate(4, 4, 4))
        ),
    )
    gt_source = ArraySource(
        gt_key,
        Array(
            array, ArraySpec(Roi((0, 0, 0), (1000, 1000, 1000)), Coordinate(4, 4, 4))
        ),
    )

    request = BatchRequest()
    request.add(raw_key, (200, 200, 200))
    request.add(raw_resampled_key, (120, 120, 120))
    request.add(gt_key, (200, 200, 200))
    request.add(gt_resampled_key, (192, 192, 192))

    pipeline = (
        (raw_source, gt_source)
        + MergeProvider()
        + Resample(raw_key, Coordinate((8, 8, 8)), raw_resampled_key)
        + Resample(
            gt_key, Coordinate((2, 2, 2)), gt_resampled_key, interp_order=0
        )  # Test upsampling, without interpolation
    )

    with build(pipeline):
        batch = pipeline.request_batch(request)

    for array_key, array in batch.arrays.items():
        # assert that pixels encode their position for supposedly unaltered
        # arrays
        if array_key in [raw_key, gt_key]:
            # the z,y,x coordinates of the ROI
            roi = array.spec.roi / 4
            meshgrids = np.meshgrid(
                range(roi.get_begin()[0], roi.get_end()[0]),
                range(roi.get_begin()[1], roi.get_end()[1]),
                range(roi.get_begin()[2], roi.get_end()[2]),
                indexing="ij",
            )
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]
            data = np.stack([data, data, data], axis=0)

            assert np.array_equal(array.data, data), str(array_key)

        elif array_key == raw_resampled_key:
            # Note: First assert averages over the voxels in the raw roi:
            # (40:48, 40:48, 40:48), values of [30,31,31,32,31,32,32,33], the average of
            # which is 31.5. Casting to an integer, in this case, rounds down, resulting
            # in 31.
            assert (
                array.data[0, 0, 0, 0] == 31
            ), f"RAW_RESAMPLED[0,0,0]: {array.data[0,0,0]} does not equal expected: 31"
            assert (
                array.data[0, 1, 0, 0] == 33
            ), f"RAW_RESAMPLED[1,0,0]: {array.data[1,0,0]} does not equal expected: 33"

        elif array_key == gt_resampled_key:
            # Note: GT_LABELS_RESAMPLED is shifted a full pixel in from each side of original array to pad upsampling
            assert (
                array.data[0, 0, 0, 0] == 3
            ), f"GT_LABELS_RESAMPLED[0,0,0]: {array.data[0,0,0]} does not equal expected: 0"
            assert (
                array.data[0, 1, 0, 0] == 3
            ), f"GT_LABELS_RESAMPLED[1,0,0]: {array.data[1,0,0]} does not equal expected: 0"
            assert (
                array.data[0, 2, 0, 0] == 4
            ), f"GT_LABELS_RESAMPLED[2,0,0]: {array.data[2,0,0]} does not equal expected: 1"
            assert (
                array.data[0, 3, 0, 0] == 4
            ), f"GT_LABELS_RESAMPLED[3,0,0]: {array.data[3,0,0]} does not equal expected: 1"

        else:
            assert False, "unexpected array type"
