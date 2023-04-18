from gunpowder import *
from gunpowder.contrib import PrepareMalis
import numpy as np
from .provider_test import ProviderTest


class ExampleSourcePrepareMalis(BatchProvider):
    def setup(self):
        self.provides(
            ArrayKeys.GT_LABELS,
            ArraySpec(
                roi=Roi((0, 0, 0), (90, 90, 90)),
                voxel_size=(1, 1, 1),
                interpolatable=False,
            ),
        )
        self.provides(
            ArrayKeys.GT_IGNORE,
            ArraySpec(
                roi=Roi((0, 0, 0), (90, 90, 90)),
                voxel_size=(1, 1, 1),
                interpolatable=False,
            ),
        )

    def provide(self, request):
        batch = Batch()

        if ArrayKeys.GT_LABELS in request:
            gt_labels_roi = request[ArrayKeys.GT_LABELS].roi
            gt_labels_shape = gt_labels_roi.shape

            data_labels = np.ones(gt_labels_shape)
            data_labels[gt_labels_shape[0] // 2 :, :, :] = 2
            spec = self.spec[ArrayKeys.GT_LABELS].copy()
            spec.roi = gt_labels_roi

            batch.arrays[ArrayKeys.GT_LABELS] = Array(data_labels, spec)

        if ArrayKeys.GT_IGNORE in request:
            gt_ignore_roi = request[ArrayKeys.GT_IGNORE].roi
            gt_ignore_shape = gt_ignore_roi.shape

            data_gt_ignore = np.ones(gt_ignore_shape)
            data_gt_ignore[:, gt_ignore_shape[1] // 6 :, :] = 0
            spec = self.spec[ArrayKeys.GT_IGNORE].copy()
            spec.roi = gt_ignore_roi

            batch.arrays[ArrayKeys.GT_IGNORE] = Array(data_gt_ignore, spec)

        return batch


class TestPrepareMalis(ProviderTest):
    def test_output(self):
        ArrayKey("MALIS_COMP_LABEL")

        pipeline_with_ignore = ExampleSourcePrepareMalis() + PrepareMalis(
            ArrayKeys.GT_LABELS,
            ArrayKeys.MALIS_COMP_LABEL,
            ignore_array_key=ArrayKeys.GT_IGNORE,
        )
        pipeline_without_ignore = ExampleSourcePrepareMalis() + PrepareMalis(
            ArrayKeys.GT_LABELS,
            ArrayKeys.MALIS_COMP_LABEL,
        )

        # test that MALIS_COMP_LABEL not in batch if not in request
        with build(pipeline_with_ignore):
            request = BatchRequest()
            request.add(ArrayKeys.GT_LABELS, (90, 90, 90))
            request.add(ArrayKeys.GT_IGNORE, (90, 90, 90))

            batch = pipeline_with_ignore.request_batch(request)

            # test if array added to batch
            self.assertTrue(ArrayKeys.MALIS_COMP_LABEL not in batch.arrays)

        # test usage with gt_ignore
        with build(pipeline_with_ignore):
            request = BatchRequest()
            request.add(ArrayKeys.GT_LABELS, (90, 90, 90))
            request.add(ArrayKeys.GT_IGNORE, (90, 90, 90))
            request.add(ArrayKeys.MALIS_COMP_LABEL, (90, 90, 90))

            batch = pipeline_with_ignore.request_batch(request)

            # test if array added to batch
            self.assertTrue(ArrayKeys.MALIS_COMP_LABEL in batch.arrays)

            # test if gt_ignore considered for gt_neg_pass ([0, ...]) and not for gt_pos_pass ([1, ...])
            ignored_locations = np.where(batch.arrays[ArrayKeys.GT_IGNORE].data == 0)
            # gt_neg_pass
            self.assertTrue(
                (
                    batch.arrays[ArrayKeys.MALIS_COMP_LABEL].data[0, ...][
                        ignored_locations
                    ]
                    == 3
                ).all()
            )
            self.assertFalse(
                (
                    np.array_equal(
                        batch.arrays[ArrayKeys.MALIS_COMP_LABEL].data[0, ...],
                        batch.arrays[ArrayKeys.GT_LABELS].data,
                    )
                )
            )
            # gt_pos_pass
            self.assertFalse(
                (
                    batch.arrays[ArrayKeys.MALIS_COMP_LABEL].data[1, ...][
                        ignored_locations
                    ]
                    == 3
                ).all()
            )
            self.assertTrue(
                (
                    np.array_equal(
                        batch.arrays[ArrayKeys.MALIS_COMP_LABEL].data[1, ...],
                        batch.arrays[ArrayKeys.GT_LABELS].data,
                    )
                )
            )

            # Test ignore without requesting ignore array
            request = BatchRequest()
            request.add(ArrayKeys.GT_LABELS, (90, 90, 90))
            request.add(ArrayKeys.MALIS_COMP_LABEL, (90, 90, 90))

            batch = pipeline_with_ignore.request_batch(request)

            # test if array added to batch
            self.assertTrue(ArrayKeys.MALIS_COMP_LABEL in batch.arrays)

            # gt_neg_pass
            self.assertTrue(
                (
                    batch.arrays[ArrayKeys.MALIS_COMP_LABEL].data[0, ...][
                        ignored_locations
                    ]
                    == 3
                ).all()
            )
            self.assertFalse(
                (
                    np.array_equal(
                        batch.arrays[ArrayKeys.MALIS_COMP_LABEL].data[0, ...],
                        batch.arrays[ArrayKeys.GT_LABELS].data,
                    )
                )
            )
            # gt_pos_pass
            self.assertFalse(
                (
                    batch.arrays[ArrayKeys.MALIS_COMP_LABEL].data[1, ...][
                        ignored_locations
                    ]
                    == 3
                ).all()
            )
            self.assertTrue(
                (
                    np.array_equal(
                        batch.arrays[ArrayKeys.MALIS_COMP_LABEL].data[1, ...],
                        batch.arrays[ArrayKeys.GT_LABELS].data,
                    )
                )
            )

        # test usage without gt_ignore
        with build(pipeline_without_ignore):
            request = BatchRequest()
            request.add(ArrayKeys.GT_LABELS, (90, 90, 90))
            request.add(ArrayKeys.MALIS_COMP_LABEL, (90, 90, 90))

            batch = pipeline_without_ignore.request_batch(request)

            # test if array added to batch
            self.assertTrue(ArrayKeys.MALIS_COMP_LABEL in batch.arrays)

            # test if gt_ignore considered for gt_neg_pass ([0, ;;;]) and not for gt_pos_pass ([1, ...])
            # gt_neg_pass
            self.assertTrue(
                (
                    np.array_equal(
                        batch.arrays[ArrayKeys.MALIS_COMP_LABEL].data[0, ...],
                        batch.arrays[ArrayKeys.GT_LABELS].data,
                    )
                )
            )
            # gt_pos_pass
            self.assertTrue(
                (
                    np.array_equal(
                        batch.arrays[ArrayKeys.MALIS_COMP_LABEL].data[1, ...],
                        batch.arrays[ArrayKeys.GT_LABELS].data,
                    )
                )
            )
