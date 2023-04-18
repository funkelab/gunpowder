from .helper_sources import ArraySource

from gunpowder import *
import numpy as np


def test_output():
    raw = ArrayKey("RAW")
    raw_downsampled = ArrayKey("RAW_DOWNSAMPLED")
    gt = ArrayKey("GT")
    gt_downsampled = ArrayKey("GT_LABELS_DOWNSAMPLED")

    request = BatchRequest()
    request.add(raw, (200, 200, 200))
    request.add(raw_downsampled, (120, 120, 120))
    request.add(gt, (200, 200, 200))
    request.add(gt_downsampled, (200, 200, 200))

    meshgrids = np.meshgrid(range(0, 250), range(0, 250), range(0, 250), indexing="ij")
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
        + DownSample(raw, 2, raw_downsampled)
        + DownSample(gt, (2, 2, 2), gt_downsampled)
    )

    with build(pipeline):
        batch = pipeline.request_batch(request)

    for array_key, array in batch.arrays.items():
        # assert that pixels encode their position for supposedly unaltered
        # arrays
        if array_key in [raw, gt]:
            # the z,y,x coordinates of the ROI
            roi = array.spec.roi / 4
            meshgrids = np.meshgrid(
                range(roi.begin[0], roi.end[0]),
                range(roi.begin[1], roi.end[1]),
                range(roi.begin[2], roi.end[2]),
                indexing="ij",
            )
            data = meshgrids[0] + meshgrids[1] + meshgrids[2]

            if array_key == raw:
                assert np.array_equal(array.data[0], data), str(array_key)
            else:
                assert np.array_equal(array.data, data), str(array_key)

        elif array_key == raw_downsampled:
            assert array.data[0, 0, 0, 0] == 30
            assert array.data[1, 1, 0, 0] == 32

        elif array_key == gt_downsampled:
            assert array.data[0, 0, 0] == 0
            assert array.data[1, 0, 0] == 2

        else:
            assert False, "unexpected array type"
