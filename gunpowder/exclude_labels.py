import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from batch_filter import BatchFilter

class ExcludeLabels(BatchFilter):

    def __init__(self, labels, include_context, background_value=0):
        self.labels = set(labels)
        self.include_context = include_context
        self.background_value = background_value

    def process(self, batch):

        # 0 marks included regions (to be used directly with distance transform 
        # later)
        include_mask = np.ones(batch.gt.shape)

        for label in np.unique(batch.gt):
            if label in self.labels:
                batch.gt[batch.gt==label] = self.background_value
            else:
                include_mask[batch.gt==label] = 0

        distance_to_include = distance_transform_edt(include_mask, sampling=batch.spec.resolution)
        print("ExcludeLabels: max distance to foreground is " + str(distance_to_include.max()))

        # 1 marks included regions, plus a context area around them
        include_mask = distance_to_include<self.include_context

        if batch.gt_mask is not None:
            batch.gt_mask &= include_mask
        else:
            batch.gt_mask = include_mask
