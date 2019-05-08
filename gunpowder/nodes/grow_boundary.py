import numpy as np
from scipy import ndimage

from .batch_filter import BatchFilter
from gunpowder.array import Array

class GrowBoundary(BatchFilter):
    '''Grow a boundary between regions in a label array. Does not grow at the
    border of the batch or an optionally provided mask.

    Args:

        labels (:class:`ArrayKey`):

            The array containing labels.

        mask (:class:`ArrayKey`, optional):

            A mask indicating unknown regions. This is to avoid boundaries to
            grow between labelled and unknown regions.

        steps (``int``, optional):

            Number of voxels (not world units!) to grow.

        background (``int``, optional):

            The label to assign to the boundary voxels.

        only_xy (``bool``, optional):

            Do not grow a boundary in the z direction.
    '''

    def __init__(self, labels, mask=None, steps=1, background=0, only_xy=False):
        self.labels = labels
        self.mask = mask
        self.steps = steps
        self.background = background
        self.only_xy = only_xy

    def process(self, batch, request):

        gt = batch.arrays[self.labels]
        gt_mask = None if not self.mask else batch.arrays[self.mask]

        if gt_mask is not None:

            # grow only in area where mask and gt are defined
            crop = gt_mask.spec.roi.intersect(gt.spec.roi)

            if crop is None:
                raise RuntimeError("GT_LABELS %s and GT_MASK %s ROIs don't intersect."%(gt.spec.roi,gt_mask.spec.roi))
            voxel_size = self.spec[self.labels].voxel_size
            crop_in_gt = (crop.shift(-gt.spec.roi.get_offset())/voxel_size).get_bounding_box()
            crop_in_gt_mask = (crop.shift(-gt_mask.spec.roi.get_offset())/voxel_size).get_bounding_box()

            self.__grow(gt.data[crop_in_gt], gt_mask.data[crop_in_gt_mask], self.only_xy)

        else:

            self.__grow(gt.data, only_xy=self.only_xy)

    def __grow(self, gt, gt_mask=None, only_xy=False):
        if gt_mask is not None:
            assert gt.shape == gt_mask.shape, "GT_LABELS and GT_MASK do not have the same size."

        if only_xy:
            assert len(gt.shape) == 3
            for z in range(gt.shape[0]):
                self.__grow(gt[z], None if gt_mask is None else gt_mask[z])
            return

        # get all foreground voxels by erosion of each component
        foreground = np.zeros(shape=gt.shape, dtype=np.bool)
        masked = None
        if gt_mask is not None:
            masked = np.equal(gt_mask, 0)
        for label in np.unique(gt):
            if label == self.background:
                continue
            label_mask = gt==label
            # Assume that masked out values are the same as the label we are
            # eroding in this iteration. This ensures that at the boundary to
            # a masked region the value blob is not shrinking.
            if masked is not None:
                label_mask = np.logical_or(label_mask, masked)
            eroded_label_mask = ndimage.binary_erosion(label_mask, iterations=self.steps, border_value=1)
            foreground = np.logical_or(eroded_label_mask, foreground)

        # label new background
        background = np.logical_not(foreground)
        gt[background] = self.background
