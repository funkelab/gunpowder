import copy
import logging
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.ext import malis
from gunpowder.volume import Volume, VolumeType

logger = logging.getLogger(__name__)

class AddGtAffinities(BatchFilter):

    def __init__(self, affinity_neighborhood):

        self.affinity_neighborhood = affinity_neighborhood

        dims = self.affinity_neighborhood.shape[1]
        self.padding_neg = tuple(
                min([0] + [a[d] for a in self.affinity_neighborhood])
                for d in range(dims)
        )

        self.padding_pos = tuple(
                max([0] + [a[d] for a in self.affinity_neighborhood])
                for d in range(dims)
        )

        logger.debug("padding neg: " + str(self.padding_neg))
        logger.debug("padding pos: " + str(self.padding_pos))

        self.skip_next = False

    def prepare(self, batch_spec):

        # do nothing if no gt affinities were requested
        if not VolumeType.GT_AFFINITIES in batch_spec.with_volumes:
            self.skip_next = True
            return

        # remember requested output shape
        self.request_output_roi = copy.deepcopy(batch_spec.output_roi)

        dims = batch_spec.output_roi.dims()

        logger.debug("downstream output ROI: " + str(self.request_output_roi))

        # shift output ROI by padding_neg
        batch_spec.output_roi = batch_spec.output_roi.shift(self.padding_neg)
        # increase shape
        shape = batch_spec.output_roi.get_shape()
        shape = tuple(shape[d] - self.padding_neg[d] + self.padding_pos[d] for d in range(dims))
        batch_spec.output_roi.set_shape(shape)

        logger.debug("upstream output ROI: " + str(batch_spec.output_roi))

    def process(self, batch):

        # do nothing if no gt affinities were requested
        if self.skip_next:
            self.skip_next = False
            return

        # do nothing if gt affinities are already present
        if VolumeType.GT_AFFINITIES in batch.volumes:
            logger.warning("batch already contains affinities, skipping")
            return

        logger.debug("computing ground-truth affinities from labels")
        gt_affinities = Volume(
                malis.seg_to_affgraph(
                        batch.volumes[VolumeType.GT_LABELS].data.astype(np.int32),
                        self.affinity_neighborhood
                ).astype(np.float32),
                interpolate=False)
        batch.volumes[VolumeType.GT_AFFINITIES] = gt_affinities

        # crop to original output ROI
        offset = self.request_output_roi.get_offset()
        shift = tuple(-offset[d]-self.padding_neg[d] for d in range(batch.spec.output_roi.dims()))
        crop_roi = self.request_output_roi.shift(shift)
        crop = crop_roi.get_bounding_box()

        logger.debug("cropping with " + str(crop))

        gt_affinities.data = gt_affinities.data[(slice(None),)+crop]
        for volume_type in [VolumeType.GT_LABELS, VolumeType.GT_MASK, VolumeType.GT_IGNORE]:
            if volume_type in batch.volumes:
                volume = batch.volumes[volume_type]
                volume.data = volume.data[crop]
        batch.spec.output_roi = batch.spec.output_roi.shift(crop_roi.get_offset())
        batch.spec.output_roi.set_shape(crop_roi.get_shape())

        logger.debug("reset output ROI to " + str(batch.spec.output_roi))

        batch.affinity_neighborhood = self.affinity_neighborhood
