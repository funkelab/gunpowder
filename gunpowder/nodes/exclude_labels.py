import logging
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

from .batch_filter import BatchFilter
from gunpowder.volume import Volume, VolumeTypes

logger = logging.getLogger(__name__)

class ExcludeLabels(BatchFilter):
    '''Excludes several labels from the ground-truth.

    The labels will be replaced by background_value. The GT_IGNORE mask will be 
    set to 0 for the excluded locations that are further than ignore_mask_erode 
    away from not excluded locations.
    '''

    def __init__(self, labels, ignore_mask_erode, background_value=0):
        '''
        Args:
            labels: List of IDs to exclude from the ground-truth.
            ignore_mask_erode: By how much (in world units) to erode the ignore mask.
            background_value: Value to replace excluded IDs.
        '''
        self.labels = set(labels)
        self.ignore_mask_erode = ignore_mask_erode
        self.background_value = background_value

    def setup(self):

        assert VolumeTypes.GT_LABELS in self.spec, "ExcludeLabels can only be used if GT_LABELS is provided upstream."
        self.provides(VolumeTypes.GT_IGNORE, self.spec[VolumeTypes.GT_LABELS])

    def prepare(self, request):

        # we add it, don't request upstream
        if VolumeTypes.GT_IGNORE in request:
            del request[VolumeTypes.GT_IGNORE]

    def process(self, batch, request):

        gt = batch.volumes[VolumeTypes.GT_LABELS]

        # 0 marks included regions (to be used directly with distance transform 
        # later)
        include_mask = np.ones(gt.data.shape)

        gt_labels = np.unique(gt.data)
        logger.debug("batch contains GT labels: " + str(gt_labels))
        for label in gt_labels:
            if label in self.labels:
                logger.debug("excluding label " + str(label))
                gt.data[gt.data==label] = self.background_value
            else:
                include_mask[gt.data==label] = 0

        voxel_size = self.spec[VolumeTypes.GT_LABELS].voxel_size
        distance_to_include = distance_transform_edt(include_mask, sampling=voxel_size)
        logger.debug("max distance to foreground is " + str(distance_to_include.max()))

        # 1 marks included regions, plus a context area around them
        include_mask = distance_to_include<self.ignore_mask_erode

        if VolumeTypes.GT_IGNORE in request:
            # include mask was computed on GT_LABELS ROI, we need to copy it to 
            # the requested GT_IGNORE ROI
            gt_ignore_roi = request[VolumeTypes.GT_IGNORE].roi
        else:
            gt_ignore_roi = gt.spec.roi

        intersection = gt.spec.roi.intersect(gt_ignore_roi)
        intersection_in_gt = intersection - gt.spec.roi.get_offset()
        intersection_in_gt_ignore = intersection - gt_ignore_roi.get_offset()

        # to voxel coordinates
        intersection_in_gt //= voxel_size
        intersection_in_gt_ignore //= voxel_size

        gt_ignore = np.zeros((gt_ignore_roi//voxel_size).get_shape(), dtype=np.uint8)
        gt_ignore[intersection_in_gt_ignore.get_bounding_box()] = include_mask[intersection_in_gt.get_bounding_box()]

        spec = self.spec[VolumeTypes.GT_LABELS].copy()
        spec.roi = gt_ignore_roi
        spec.dtype = np.uint8
        batch.volumes[VolumeTypes.GT_IGNORE] = Volume(gt_ignore, spec)
