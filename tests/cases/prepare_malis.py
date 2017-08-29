from gunpowder import *
import numpy as np
from .provider_test import ProviderTest

class TestSourcePrepareMalis(BatchProvider):

    def setup(self):

        self.provides(
            VolumeTypes.GT_LABELS,
            VolumeSpec(
                roi=Roi((0, 0, 0), (90, 90, 90)),
                voxel_size=(1, 1, 1),
                interpolatable=False))
        self.provides(
            VolumeTypes.GT_IGNORE,
            VolumeSpec(
                roi=Roi((0, 0, 0), (90, 90, 90)),
                voxel_size=(1, 1, 1),
                interpolatable=False))

    def provide(self, request):

        batch = Batch()

        if VolumeTypes.GT_LABELS in request:

            gt_labels_roi   = request[VolumeTypes.GT_LABELS].roi
            gt_labels_shape = gt_labels_roi.get_shape()

            data_labels = np.ones(gt_labels_shape)
            data_labels[gt_labels_shape[0]//2:, :, :] = 2
            spec = self.spec[VolumeTypes.GT_LABELS].copy()
            spec.roi = gt_labels_roi

            batch.volumes[VolumeTypes.GT_LABELS] = Volume(
                data_labels,
                spec)

        if VolumeTypes.GT_IGNORE in request:

            gt_ignore_roi   = request[VolumeTypes.GT_IGNORE].roi
            gt_ignore_shape = gt_ignore_roi.get_shape()

            data_gt_ignore = np.ones(gt_ignore_shape)
            data_gt_ignore[:, gt_ignore_shape[1]//6:, :] = 0
            spec = self.spec[VolumeTypes.GT_IGNORE].copy()
            spec.roi = gt_ignore_roi

            batch.volumes[VolumeTypes.GT_IGNORE] = Volume(
                data_gt_ignore,
                spec)

        return batch


class TestPrepareMalis(ProviderTest):

    def test_output(self):

        pipeline = TestSourcePrepareMalis() + PrepareMalis()

        # test that MALIS_COMP_LABEL not in batch if not in request
        with build(pipeline):
            request = BatchRequest()
            request.add(VolumeTypes.GT_LABELS, (90, 90, 90))
            request.add(VolumeTypes.GT_IGNORE, (90, 90, 90))

            batch = pipeline.request_batch(request)

            # test if volume added to batch
            self.assertTrue(VolumeTypes.MALIS_COMP_LABEL not in batch.volumes)

        # test usage with gt_ignore
        with build(pipeline):

            request = BatchRequest()
            request.add(VolumeTypes.GT_LABELS, (90, 90, 90))
            request.add(VolumeTypes.GT_IGNORE, (90, 90, 90))
            request.add(VolumeTypes.MALIS_COMP_LABEL, (90, 90, 90))

            batch = pipeline.request_batch(request)

            # test if volume added to batch
            self.assertTrue(VolumeTypes.MALIS_COMP_LABEL in batch.volumes)

            # test if gt_ignore considered for gt_neg_pass ([0, ...]) and not for gt_pos_pass ([1, ...])
            ignored_locations = np.where(batch.volumes[VolumeTypes.GT_IGNORE].data == 0)
            # gt_neg_pass
            self.assertTrue((batch.volumes[VolumeTypes.MALIS_COMP_LABEL].data[0,...][ignored_locations] == 3).all())
            self.assertFalse((np.array_equal(batch.volumes[VolumeTypes.MALIS_COMP_LABEL].data[0, ...],
                                            batch.volumes[VolumeTypes.GT_LABELS].data)))
            # gt_pos_pass
            self.assertFalse((batch.volumes[VolumeTypes.MALIS_COMP_LABEL].data[1,...][ignored_locations] == 3).all())
            self.assertTrue((np.array_equal(batch.volumes[VolumeTypes.MALIS_COMP_LABEL].data[1, ...],
                                            batch.volumes[VolumeTypes.GT_LABELS].data)))

        # test usage without gt_ignore
        with build(pipeline):

            request = BatchRequest()
            request.add(VolumeTypes.GT_LABELS, (90, 90, 90))
            request.add(VolumeTypes.MALIS_COMP_LABEL, (90, 90, 90))

            batch = pipeline.request_batch(request)

            # test if volume added to batch
            self.assertTrue(VolumeTypes.MALIS_COMP_LABEL in batch.volumes)

            # test if gt_ignore considered for gt_neg_pass ([0, ;;;]) and not for gt_pos_pass ([1, ...])
            # gt_neg_pass
            self.assertTrue((np.array_equal(batch.volumes[VolumeTypes.MALIS_COMP_LABEL].data[0, ...],
                                            batch.volumes[VolumeTypes.GT_LABELS].data)))
            # gt_pos_pass
            self.assertTrue((np.array_equal(batch.volumes[VolumeTypes.MALIS_COMP_LABEL].data[1, ...],
                                            batch.volumes[VolumeTypes.GT_LABELS].data)))
