import copy
import logging
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.coordinate import Coordinate
from gunpowder.ext import malis
from gunpowder.volume import Volume, VolumeType

logger = logging.getLogger(__name__)

class AddGtAffinities(BatchFilter):

    def __init__(self, affinity_neighborhood):

        self.affinity_neighborhood = affinity_neighborhood

        dims = self.affinity_neighborhood.shape[1]
        self.padding_neg = Coordinate(
                min([0] + [a[d] for a in self.affinity_neighborhood])
                for d in range(dims)
        )

        self.padding_pos = Coordinate(
                max([0] + [a[d] for a in self.affinity_neighborhood])
                for d in range(dims)
        )

        logger.debug("padding neg: " + str(self.padding_neg))
        logger.debug("padding pos: " + str(self.padding_pos))

        self.skip_next = False

    def prepare(self, request):

        # do nothing if no gt affinities were requested
        if not VolumeType.GT_AFFINITIES in request.volumes:
            logger.warn("no GT_AFFINITIES requested, will do nothing")
            self.skip_next = True
            return

        assert VolumeType.GT_LABELS in request.volumes, "AddGtAffinities can only be used if you request GT_LABELS"

        del request.volumes[VolumeType.GT_AFFINITIES]

        gt_labels_roi = request.volumes[VolumeType.GT_LABELS]
        logger.debug("downstream GT_LABELS request: " + str(gt_labels_roi))

        # remember requested GT_LABELS ROI
        self.gt_labels_roi = copy.deepcopy(gt_labels_roi)

        # shift GT_LABELS ROI by padding_neg
        gt_labels_roi = gt_labels_roi.shift(self.padding_neg)
        # increase shape
        shape = gt_labels_roi.get_shape()
        shape = shape - self.padding_neg + self.padding_pos
        gt_labels_roi.set_shape(shape)
        request.volumes[VolumeType.GT_LABELS] = gt_labels_roi

        logger.debug("upstream GT_LABELS request: " + str(gt_labels_roi))

    def process(self, batch):

        # do nothing if no gt affinities were requested
        if self.skip_next:
            self.skip_next = False
            return

        logger.debug("computing ground-truth affinities from labels")
        gt_affinities = malis.seg_to_affgraph(
                batch.volumes[VolumeType.GT_LABELS].data.astype(np.int32),
                self.affinity_neighborhood
        ).astype(np.float32)

        # crop to original GT_LABELS ROI
        offset = self.gt_labels_roi.get_offset()
        shift = -offset - self.padding_neg
        crop_roi = self.gt_labels_roi.shift(shift)
        crop = crop_roi.get_bounding_box()

        logger.debug("cropping with " + str(crop))
        gt_affinities = gt_affinities[(slice(None),)+crop]

        logger.debug("reset GT_LABELS ROI to " + str(self.gt_labels_roi))
        batch.volumes[VolumeType.GT_LABELS].data = batch.volumes[VolumeType.GT_LABELS].data[crop]
        batch.volumes[VolumeType.GT_LABELS].roi = self.gt_labels_roi
        batch.volumes[VolumeType.GT_AFFINITIES] = Volume(
                gt_affinities,
                self.gt_labels_roi, 
                batch.volumes[VolumeType.GT_LABELS].resolution,
                interpolate=False)
        batch.affinity_neighborhood = self.affinity_neighborhood
