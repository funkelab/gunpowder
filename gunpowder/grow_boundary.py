from scipy import ndimage
import numpy as np
from batch_filter import BatchFilter

class GrowBoundary(BatchFilter):
    '''Grow a boundary between regions. Does not grow at the border of the batch 
    or the mask (if provided).
    '''

    def __init__(self, steps=1, background=0, only_xy=False):
        self.steps = steps
        self.background = background
        self.only_xy = only_xy

    def process(self, batch):
        self.__grow(batch.gt, batch.gt_mask, self.only_xy)

    def __grow(self, gt, gt_mask=None, only_xy=False):

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
