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

        gt_labels(:class:``VolumeType``, optional): The volume type to
            read the labels from. Defaults to ``GT_LABELS``.

        gt_labels_mask(:class:``VolumeType``, optional): The volume type to use
            as a mask for ``gt_labels``. Affinities connecting at least one
            masked out label will be masked out in ``gt_affinities_mask``. If
            not given, ``GT_AFFINITIES_MASK`` will contain ones everywhere (if
            requested).

        gt_affinities(:class:``VolumeType``, optional): The volume type
            to generate containing the affinities. Defaults to
            ``GT_AFFINITIES``.

        gt_affinities_mask(:class:``VolumeType``, optional): The volume type to
            generate containing the affinitiy mask, as derived from parameter
            ``gt_labels_mask``. Defaults to ``GT_AFFINITIES_MASK``.
    '''

    def __init__(
            self,
            affinity_neighborhood,
            gt_labels=None,
            gt_labels_mask=None,
            gt_affinities=None,
            gt_affinities_mask=None):

        if gt_labels is None:
            gt_labels = VolumeTypes.GT_LABELS
        if gt_affinities is None:
            gt_affinities = VolumeTypes.GT_AFFINITIES
        if gt_affinities_mask is None:
            gt_affinities_mask = VolumeTypes.GT_AFFINITIES_MASK

        self.affinity_neighborhood = np.array(affinity_neighborhood)
        self.gt_labels = gt_labels
        self.gt_labels_mask = gt_labels_mask
        self.gt_affinities = gt_affinities
        self.gt_affinities_mask = gt_affinities_mask

        self.skip_next = False

    def setup(self):

        assert self.gt_labels in self.spec, "Upstream does not provide %s needed by AddGtAffinities"%self.gt_labels

        voxel_size = self.spec[self.gt_labels].voxel_size

        dims = self.affinity_neighborhood.shape[1]
        self.padding_neg = Coordinate(
                min([0] + [a[d] for a in self.affinity_neighborhood])
                for d in range(dims)
        )*voxel_size

        self.padding_pos = Coordinate(
                max([0] + [a[d] for a in self.affinity_neighborhood])
                for d in range(dims)
        )*voxel_size

        logger.debug("padding neg: " + str(self.padding_neg))
        logger.debug("padding pos: " + str(self.padding_pos))

        spec = self.spec[self.gt_labels].copy()
        if spec.roi is not None:
            spec.roi = spec.roi.grow(self.padding_neg, -self.padding_pos)
        spec.dtype = np.float32

        self.provides(self.gt_affinities, spec)
        self.provides(self.gt_affinities_mask, spec)


    def prepare(self, request):

        # do nothing if no gt affinities were requested
        if not self.gt_affinities in request:
            logger.warn("no affinites requested, will do nothing")
            self.skip_next = True
            return

        del request[self.gt_affinities]
        if self.gt_affinities_mask in request:
            del request[self.gt_affinities_mask]

        if self.gt_labels_mask:
            assert (
                request[self.gt_labels].roi ==
                request[self.gt_labels_mask].roi),(
                "requested GT label roi %s and GT label mask roi %s are not "
                "the same."%(
                    request[self.gt_labels].roi,
                    request[self.gt_labels_mask].roi))

        gt_labels_roi = request[self.gt_labels].roi
        logger.debug("downstream %s request: "%self.gt_labels + str(gt_labels_roi))

        # grow labels ROI to accomodate padding
        gt_labels_roi = gt_labels_roi.grow(-self.padding_neg, self.padding_pos)
        request[self.gt_labels].roi = gt_labels_roi

        # same for label mask
        if self.gt_labels_mask:
            request[self.gt_labels_mask].roi = gt_labels_roi.copy()

        logger.debug("upstream %s request: "%self.gt_labels + str(gt_labels_roi))

    def process(self, batch, request):

        # do nothing if no gt affinities were requested
        if self.skip_next:
            self.skip_next = False
            return

        gt_labels_roi = request[self.gt_labels].roi

        logger.debug("computing ground-truth affinities from labels")
        gt_affinities = malis.seg_to_affgraph(
                batch.volumes[self.gt_labels].data.astype(np.int32),
                self.affinity_neighborhood
        ).astype(np.float32)


        # crop affinities to original label ROI
        offset = gt_labels_roi.get_offset()
        shift = -offset - self.padding_neg
        crop_roi = gt_labels_roi.shift(shift)
        crop_roi /= self.spec[self.gt_labels].voxel_size
        crop = crop_roi.get_bounding_box()

        logger.debug("cropping with " + str(crop))
        gt_affinities = gt_affinities[(slice(None),)+crop]

        spec = self.spec[self.gt_affinities].copy()
        spec.roi = gt_labels_roi
        batch.volumes[self.gt_affinities] = Volume(gt_affinities, spec)

        if self.gt_affinities_mask in request:

            if self.gt_labels_mask is not None:

                logger.debug("computing ground-truth affinities mask from "
                             "labels mask")
                gt_affinities_mask = malis.seg_to_affgraph(
                        batch.volumes[self.gt_labels_mask].data.astype(np.int32),
                        self.affinity_neighborhood
                ).astype(np.float32)

            else:

                gt_affinities_mask = np.ones_like(gt_affinities)

            gt_affinities_mask = gt_affinities_mask[(slice(None),)+crop]
            batch.volumes[self.gt_affinities_mask] = Volume(gt_affinities_mask, spec)

        else:

            if self.gt_labels_mask is not None:
                logger.warning("GT labels does have a mask, but affinities "
                               "mask is not requested.")

        # crop labels to original label ROI
        batch.volumes[self.gt_labels] = batch.volumes[self.gt_labels].crop(gt_labels_roi)

        # same for label mask
        if self.gt_labels_mask:
            batch.volumes[self.gt_labels_mask] = batch.volumes[self.gt_labels_mask].crop(gt_labels_roi)

        batch.affinity_neighborhood = self.affinity_neighborhood
