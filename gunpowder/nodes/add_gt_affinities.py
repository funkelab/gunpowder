import copy
import logging
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.coordinate import Coordinate
from gunpowder.ext import malis
from gunpowder.volume import Volume, VolumeTypes

logger = logging.getLogger(__name__)

class AddGtAffinities(BatchFilter):
    '''Add a volume with affinities for a given label volume and neighborhood to 
    the batch. Affinity values are created one for each voxel and entry in the 
    neighborhood list, i.e., for each voxel and each neighbor of this voxel. 
    Values are 1 iff both labels (of the voxel and the neighbor) are equal and 
    non-zero.

    Args:

        affinity_neighborhood(list of offsets): List of offsets for the 
            affinities to consider for each voxel.

        label_volume_type(:class:``VolumeType``): The volume type to read the 
            labels from.

        affinity_volume_type(:class:``VolumeType``): The volume type to generate 
            containing the affinities.
    '''

    def __init__(self, affinity_neighborhood, label_volume_type=None, affinity_volume_type=None):

        if label_volume_type is None:
            label_volume_type = VolumeTypes.GT_LABELS
        if affinity_volume_type is None:
            affinity_volume_type = VolumeTypes.GT_AFFINITIES

        self.affinity_neighborhood = np.array(affinity_neighborhood)
        self.label_volume_type = label_volume_type
        self.affinity_volume_type = affinity_volume_type

        dims = self.affinity_neighborhood.shape[1]
        self.padding_neg = Coordinate(
                min([0] + [a[d] for a in self.affinity_neighborhood])
                for d in range(dims)
        )*self.label_volume_type.voxel_size

        self.padding_pos = Coordinate(
                max([0] + [a[d] for a in self.affinity_neighborhood])
                for d in range(dims)
        )*self.label_volume_type.voxel_size

        logger.debug("padding neg: " + str(self.padding_neg))
        logger.debug("padding pos: " + str(self.padding_pos))

        self.skip_next = False

    def setup(self):

        self.upstream_spec = self.get_upstream_provider().get_spec()
        self.spec = copy.deepcopy(self.upstream_spec)

        assert self.label_volume_type in self.spec.volumes, "AddGtAffinities can only be used if you provide %s"%self.label_volume_type
        self.spec.volumes[self.affinity_volume_type] = self.spec.volumes[self.label_volume_type]

    def get_spec(self):
        return self.spec

    def prepare(self, request):

        # do nothing if no gt affinities were requested
        if not self.affinity_volume_type in request.volumes:
            logger.warn("no affinites requested, will do nothing")
            self.skip_next = True
            return

        del request.volumes[self.affinity_volume_type]

        gt_labels_roi = request.volumes[self.label_volume_type]
        logger.debug("downstream %s request: "%self.label_volume_type + str(gt_labels_roi))

        # shift labels ROI by padding_neg
        gt_labels_roi = gt_labels_roi.shift(self.padding_neg)
        # increase shape
        shape = gt_labels_roi.get_shape()
        shape = shape - self.padding_neg + self.padding_pos
        gt_labels_roi.set_shape(shape)
        request.volumes[self.label_volume_type] = gt_labels_roi

        logger.debug("upstream %s request: "%self.label_volume_type + str(gt_labels_roi))

    def process(self, batch, request):

        # do nothing if no gt affinities were requested
        if self.skip_next:
            self.skip_next = False
            return

        gt_labels_roi = request.volumes[self.label_volume_type]

        logger.debug("computing ground-truth affinities from labels")
        gt_affinities = malis.seg_to_affgraph(
                batch.volumes[self.label_volume_type].data.astype(np.int32),
                self.affinity_neighborhood
        ).astype(np.float32)

        # crop affinities to original label ROI
        offset = gt_labels_roi.get_offset()
        shift = -offset - self.padding_neg
        crop_roi = gt_labels_roi.shift(shift)
        crop_roi /= self.label_volume_type.voxel_size
        crop = crop_roi.get_bounding_box()

        logger.debug("cropping with " + str(crop))
        gt_affinities = gt_affinities[(slice(None),)+crop]

        batch.volumes[self.affinity_volume_type] = Volume(
                gt_affinities,
                gt_labels_roi)

        # crop labels to original label ROI
        batch.volumes[self.label_volume_type] = batch.volumes[self.label_volume_type].crop(gt_labels_roi)

        batch.affinity_neighborhood = self.affinity_neighborhood
