import copy
import logging
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.coordinate import Coordinate
from gunpowder.ext import malis
from gunpowder.volume import Volume, VolumeTypes

logger = logging.getLogger(__name__)

class AddGtAffinities(BatchFilter):

    def __init__(self, affinity_neighborhood):

        self.affinity_neighborhood = affinity_neighborhood

        dims = self.affinity_neighborhood.shape[1]
        self.padding_neg = Coordinate(
                min([0] + [a[d] for a in self.affinity_neighborhood])
                for d in range(dims)
        )*VolumeTypes.GT_LABELS.voxel_size

        self.padding_pos = Coordinate(
                max([0] + [a[d] for a in self.affinity_neighborhood])
                for d in range(dims)
        )*VolumeTypes.GT_LABELS.voxel_size

        logger.debug("padding neg: " + str(self.padding_neg))
        logger.debug("padding pos: " + str(self.padding_pos))

        self.skip_next = False

    def setup(self):

        self.upstream_spec = self.get_upstream_provider().get_spec()
        self.spec = copy.deepcopy(self.upstream_spec)

        assert VolumeTypes.GT_LABELS in self.spec.volumes, "AddGtAffinities can only be used if you GT_LABELS provided"
        self.spec.volumes[VolumeTypes.GT_AFFINITIES] = self.spec.volumes[VolumeTypes.GT_LABELS]

    def get_spec(self):
        return self.spec

    def prepare(self, request):

        # do nothing if no gt affinities were requested
        if not VolumeTypes.GT_AFFINITIES in request.volumes:
            logger.warn("no GT_AFFINITIES requested, will do nothing")
            self.skip_next = True
            return

        del request.volumes[VolumeTypes.GT_AFFINITIES]

        gt_labels_roi = request.volumes[VolumeTypes.GT_LABELS]
        logger.debug("downstream GT_LABELS request: " + str(gt_labels_roi))

        # shift GT_LABELS ROI by padding_neg
        gt_labels_roi = gt_labels_roi.shift(self.padding_neg)
        # increase shape
        shape = gt_labels_roi.get_shape()
        shape = shape - self.padding_neg + self.padding_pos
        gt_labels_roi.set_shape(shape)
        request.volumes[VolumeTypes.GT_LABELS] = gt_labels_roi

        logger.debug("upstream GT_LABELS request: " + str(gt_labels_roi))

    def process(self, batch, request):

        # do nothing if no gt affinities were requested
        if self.skip_next:
            self.skip_next = False
            return

        gt_labels_roi = request.volumes[VolumeTypes.GT_LABELS]

        logger.debug("computing ground-truth affinities from labels")
        gt_affinities = malis.seg_to_affgraph(
                batch.volumes[VolumeTypes.GT_LABELS].data.astype(np.int32),
                self.affinity_neighborhood
        ).astype(np.float32)

        # crop affinities original GT_LABELS ROI
        offset = gt_labels_roi.get_offset()
        shift = -offset - self.padding_neg
        crop_roi = gt_labels_roi.shift(shift)
        crop_roi /= VolumeTypes.GT_LABELS.voxel_size
        crop = crop_roi.get_bounding_box()

        logger.debug("cropping with " + str(crop))
        gt_affinities = gt_affinities[(slice(None),)+crop]

        batch.volumes[VolumeTypes.GT_AFFINITIES] = Volume(
                gt_affinities,
                gt_labels_roi)

        # crop to original GT_LABELS ROI
        batch.volumes[VolumeTypes.GT_LABELS] = batch.volumes[VolumeTypes.GT_LABELS].crop(gt_labels_roi)

        batch.affinity_neighborhood = self.affinity_neighborhood
