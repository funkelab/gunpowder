import logging
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

from .batch_filter import BatchFilter
from gunpowder.array import Array

logger = logging.getLogger(__name__)

class ExcludeLabels(BatchFilter):
    '''Excludes several labels from the ground-truth.

    The labels will be replaced by background_value. An optional ignore mask
    will be created and set to 0 for the excluded locations that are further
    than a threshold away from not excluded locations.

    Args:

        labels (:class:`ArrayKey`):

            The array containing the labels.

        exclude (``list`` of ``int``):

            The labels to exclude from ``labels``.

        ignore_mask (:class:`ArrayKey`, optional):

            The ignore mask to create.

        ignore_mask_erode (``float``, optional):

            By how much (in world units) to erode the ignore mask.

        background_value (``int``, optional):

            Value to replace excluded IDs, defaults to 0.
    '''

    def __init__(
            self,
            labels,
            exclude,
            ignore_mask=None,
            ignore_mask_erode=0,
            background_value=0):

        self.labels = labels
        self.exclude = set(exclude)
        self.ignore_mask = ignore_mask
        self.ignore_mask_erode = ignore_mask_erode
        self.background_value = background_value

    def setup(self):

        assert self.labels in self.spec, "ExcludeLabels can only be used if GT_LABELS is provided upstream."
        if self.ignore_mask:
            self.provides(self.ignore_mask, self.spec[self.labels])

    def process(self, batch, request):

        gt = batch.arrays[self.labels]

        # 0 marks included regions (to be used directly with distance transform
        # later)
        include_mask = np.ones(gt.data.shape)

        gt_labels = np.unique(gt.data)
        logger.debug("batch contains GT labels: " + str(gt_labels))
        for label in gt_labels:
            if label in self.exclude:
                logger.debug("excluding label " + str(label))
                gt.data[gt.data==label] = self.background_value
            else:
                include_mask[gt.data==label] = 0

        # if no ignore mask is provided or requested, we are done
        if not self.ignore_mask or not self.ignore_mask in request:
            return

        voxel_size = self.spec[self.labels].voxel_size
        distance_to_include = distance_transform_edt(include_mask, sampling=voxel_size)
        logger.debug("max distance to foreground is " + str(distance_to_include.max()))

        # 1 marks included regions, plus a context area around them
        include_mask = distance_to_include<self.ignore_mask_erode

        # include mask was computed on labels ROI, we need to copy it to
        # the requested ignore_mask ROI
        gt_ignore_roi = request[self.ignore_mask].roi

        intersection = gt.spec.roi.intersect(gt_ignore_roi)
        intersection_in_gt = intersection - gt.spec.roi.get_offset()
        intersection_in_gt_ignore = intersection - gt_ignore_roi.get_offset()

        # to voxel coordinates
        intersection_in_gt //= voxel_size
        intersection_in_gt_ignore //= voxel_size

        gt_ignore = np.zeros((gt_ignore_roi//voxel_size).get_shape(), dtype=np.uint8)
        gt_ignore[intersection_in_gt_ignore.get_bounding_box()] = include_mask[intersection_in_gt.get_bounding_box()]

        spec = self.spec[self.labels].copy()
        spec.roi = gt_ignore_roi
        spec.dtype = np.uint8
        batch.arrays[self.ignore_mask] = Array(gt_ignore, spec)
