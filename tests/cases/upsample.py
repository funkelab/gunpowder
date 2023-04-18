from .helper_sources import ArraySource
from gunpowder import *
import numpy as np


def test_output():
    raw = ArrayKey("RAW")
    raw_upsampled = ArrayKey("RAW_UPSAMPLED")
    gt = ArrayKey("GT")
    gt_upsampled = ArrayKey("GT_LABELS_UPSAMPLED")

    request = BatchRequest()
    request.add(raw, (200, 200, 200))
    request.add(raw_upsampled, (124, 124, 124))
    request.add(gt, (200, 200, 200))
    request.add(gt_upsampled, (200, 200, 200))

    meshgrids = np.meshgrid(
        range(0, 1000, 4), range(0, 1000, 4), range(0, 1000, 4), indexing="ij"
    )
    data = meshgrids[0] + meshgrids[1] + meshgrids[2]

    raw_source = ArraySource(
        raw,
        Array(
            np.stack([data, data]),
            ArraySpec(roi=Roi((0, 0, 0), (1000, 1000, 1000)), voxel_size=(4, 4, 4)),
        ),
    )
    gt_source = ArraySource(
        gt,
        Array(
            data,
            ArraySpec(roi=Roi((0, 0, 0), (1000, 1000, 1000)), voxel_size=(4, 4, 4)),
        ),
    )

    pipeline = (
        (raw_source, gt_source)
        + MergeProvider()
        + UpSample(raw, 2, raw_upsampled)
        + UpSample(gt, 2, gt_upsampled)
    )

    with build(pipeline):
        batch = pipeline.request_batch(request)

    for array_key, array in batch.arrays.items():
        # assert that pixels encode their position for supposedly unaltered
        # arrays
        if array_key in [raw, gt]:
            # the z,y,x coordinates of the ROI
            meshgrids = np.meshgrid(
                range(array.spec.roi.begin[0], array.spec.roi.end[0], 4),
                range(array.spec.roi.begin[1], array.spec.roi.end[1], 4),
                range(array.spec.roi.begin[2], array.spec.roi.end[2], 4),
                indexing="ij",
            )
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            if array_key == raw:
                assert np.array_equal(array.data[0], data) and np.array_equal(
                    array.data[1], data
                ), f"{array.data, data}"
            else:
                assert np.array_equal(array.data, data), str(array_key)

        elif array_key == raw_upsampled:
            assert array.data[0, 0, 0, 0] == 108
            assert array.data[1, 1, 0, 0] == 112
            assert array.data[0, 2, 0, 0] == 112
            assert array.data[0, 3, 0, 0] == 116

        elif array_key == gt_upsampled:
            assert array.data[0, 0, 0] == 0
            assert array.data[1, 0, 0] == 0
            assert array.data[2, 0, 0] == 4
            assert array.data[3, 0, 0] == 4

        else:
            assert False, "unexpected array type"
