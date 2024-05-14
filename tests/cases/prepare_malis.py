import numpy as np

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    Batch,
    BatchProvider,
    BatchRequest,
    Roi,
    build,
)
from gunpowder.contrib import PrepareMalis


class ExampleSourcePrepareMalis(BatchProvider):
    def __init__(self, labels_key, ignore_key):
        self.labels_key = labels_key
        self.ignore_key = ignore_key

    def setup(self):
        self.provides(
            self.labels_key,
            ArraySpec(
                roi=Roi((0, 0, 0), (90, 90, 90)),
                voxel_size=(1, 1, 1),
                interpolatable=False,
            ),
        )
        self.provides(
            self.ignore_key,
            ArraySpec(
                roi=Roi((0, 0, 0), (90, 90, 90)),
                voxel_size=(1, 1, 1),
                interpolatable=False,
            ),
        )

    def provide(self, request):
        batch = Batch()

        if self.labels_key in request:
            gt_labels_roi = request[self.labels_key].roi
            gt_labels_shape = gt_labels_roi.shape

            data_labels = np.ones(gt_labels_shape)
            data_labels[gt_labels_shape[0] // 2 :, :, :] = 2
            spec = self.spec[self.labels_key].copy()
            spec.roi = gt_labels_roi

            batch.arrays[self.labels_key] = Array(data_labels, spec)

        if self.ignore_key in request:
            gt_ignore_roi = request[self.ignore_key].roi
            gt_ignore_shape = gt_ignore_roi.shape

            data_gt_ignore = np.ones(gt_ignore_shape)
            data_gt_ignore[:, gt_ignore_shape[1] // 6 :, :] = 0
            spec = self.spec[self.ignore_key].copy()
            spec.roi = gt_ignore_roi

            batch.arrays[self.ignore_key] = Array(data_gt_ignore, spec)

        return batch


def test_malis():
    malis_key = ArrayKey("MALIS_COMP_LABEL")
    labels_key = ArrayKey("LABELS")
    ignore_key = ArrayKey("GT_IGNORE")

    pipeline_with_ignore = ExampleSourcePrepareMalis(
        labels_key, ignore_key
    ) + PrepareMalis(
        labels_key,
        malis_key,
        ignore_array_key=ignore_key,
    )
    pipeline_without_ignore = ExampleSourcePrepareMalis(
        labels_key, ignore_key
    ) + PrepareMalis(
        labels_key,
        malis_key,
    )

    # test that MALIS_COMP_LABEL not in batch if not in request
    with build(pipeline_with_ignore):
        request = BatchRequest()
        request.add(labels_key, (90, 90, 90))
        request.add(ignore_key, (90, 90, 90))

        batch = pipeline_with_ignore.request_batch(request)

        # test if array added to batch
        assert malis_key not in batch.arrays

    # test usage with gt_ignore
    with build(pipeline_with_ignore):
        request = BatchRequest()
        request.add(labels_key, (90, 90, 90))
        request.add(ignore_key, (90, 90, 90))
        request.add(malis_key, (90, 90, 90))

        batch = pipeline_with_ignore.request_batch(request)

        # test if array added to batch
        assert malis_key in batch.arrays

        # test if gt_ignore considered for gt_neg_pass ([0, ...]) and not for gt_pos_pass ([1, ...])
        ignored_locations = np.where(batch.arrays[ignore_key].data == 0)
        # gt_neg_pass
        assert (batch.arrays[malis_key].data[0, ...][ignored_locations] == 3).all()
        assert not (
            np.array_equal(
                batch.arrays[malis_key].data[0, ...],
                batch.arrays[labels_key].data,
            )
        )
        # gt_pos_pass
        assert not (
            (batch.arrays[malis_key].data[1, ...][ignored_locations] == 3).all()
        )
        assert np.array_equal(
            batch.arrays[malis_key].data[1, ...],
            batch.arrays[labels_key].data,
        )

        # Test ignore without requesting ignore array
        request = BatchRequest()
        request.add(labels_key, (90, 90, 90))
        request.add(malis_key, (90, 90, 90))

        batch = pipeline_with_ignore.request_batch(request)

        # test if array added to batch
        assert malis_key in batch.arrays

        # gt_neg_pass
        assert (batch.arrays[malis_key].data[0, ...][ignored_locations] == 3).all()
        assert not (
            np.array_equal(
                batch.arrays[malis_key].data[0, ...],
                batch.arrays[labels_key].data,
            )
        )
        # gt_pos_pass
        assert not (
            (batch.arrays[malis_key].data[1, ...][ignored_locations] == 3).all()
        )
        assert np.array_equal(
            batch.arrays[malis_key].data[1, ...],
            batch.arrays[labels_key].data,
        )

    # test usage without gt_ignore
    with build(pipeline_without_ignore):
        request = BatchRequest()
        request.add(labels_key, (90, 90, 90))
        request.add(malis_key, (90, 90, 90))

        batch = pipeline_without_ignore.request_batch(request)

        # test if array added to batch
        assert malis_key in batch.arrays

        # test if gt_ignore considered for gt_neg_pass ([0, ;;;]) and not for gt_pos_pass ([1, ...])
        # gt_neg_pass
        assert np.array_equal(
            batch.arrays[malis_key].data[0, ...],
            batch.arrays[labels_key].data,
        )
        # gt_pos_pass
        assert np.array_equal(
            batch.arrays[malis_key].data[1, ...],
            batch.arrays[labels_key].data,
        )
