import logging
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.coordinate import Coordinate
from gunpowder.ext import malis
from gunpowder.array import Array

logger = logging.getLogger(__name__)

class AddAffinities(BatchFilter):
    '''Add an array with affinities for a given label array and neighborhood to 
    the batch. Affinity values are created one for each voxel and entry in the 
    neighborhood list, i.e., for each voxel and each neighbor of this voxel. 
    Values are 1 iff both labels (of the voxel and the neighbor) are equal and 
    non-zero.

    Args:

        affinity_neighborhood (``list`` of array-like):

            List of offsets for the affinities to consider for each voxel.

        labels (:class:`ArrayKey`):

            The array to read the labels from.

        affinities (:class:`ArrayKey`):

            The array to generate containing the affinities.

        labels_mask (:class:`ArrayKey`, optional):

            The array to use as a mask for ``labels``. Affinities connecting at
            least one masked out label will be masked out in
            ``affinities_mask``. If not given, ``affinities_mask`` will contain
            ones everywhere (if requested).

        unlabelled (:class:`ArrayKey`, optional):

            A binary array to indicate unlabelled areas with 0. Affinities from
            labelled to unlabelled voxels are set to 0, affinities between
            unlabelled voxels are masked out (they will not be used for
            training).

        affinities_mask (:class:`ArrayKey`, optional):

            The array to generate containing the affinitiy mask, as derived
            from parameter ``labels_mask``.
    '''

    def __init__(
            self,
            affinity_neighborhood,
            labels,
            affinities,
            labels_mask=None,
            unlabelled=None,
            affinities_mask=None):

        self.affinity_neighborhood = np.array(affinity_neighborhood)
        self.labels = labels
        self.unlabelled = unlabelled
        self.labels_mask = labels_mask
        self.affinities = affinities
        self.affinities_mask = affinities_mask

    def setup(self):

        assert self.labels in self.spec, (
            "Upstream does not provide %s needed by "
            "AddAffinities"%self.labels)

        voxel_size = self.spec[self.labels].voxel_size

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

        spec = self.spec[self.labels].copy()
        if spec.roi is not None:
            spec.roi = spec.roi.grow(self.padding_neg, -self.padding_pos)
        spec.dtype = np.float32

        self.provides(self.affinities, spec)
        if self.affinities_mask:
            self.provides(self.affinities_mask, spec)
        self.enable_autoskip()

    def prepare(self, request):

        if self.labels_mask:
            assert (
                request[self.labels].roi ==
                request[self.labels_mask].roi),(
                "requested GT label roi %s and GT label mask roi %s are not "
                "the same."%(
                    request[self.labels].roi,
                    request[self.labels_mask].roi))

        if self.unlabelled:
            assert (
                request[self.labels].roi ==
                request[self.unlabelled].roi),(
                "requested GT label roi %s and GT unlabelled mask roi %s are not "
                "the same."%(
                    request[self.labels].roi,
                    request[self.unlabelled].roi))

        if self.labels not in request:
            request[self.labels] = request[self.affinities].copy()

        labels_roi = request[self.labels].roi
        context_roi = request[self.affinities].roi.grow(
            -self.padding_neg,
            self.padding_pos)

        # grow labels ROI to accomodate padding
        labels_roi = labels_roi.union(context_roi)
        request[self.labels].roi = labels_roi

        # same for label mask
        if self.labels_mask and self.labels_mask in request:
            request[self.labels_mask].roi = \
                request[self.labels_mask].roi.union(context_roi)

        # and unlabelled mask
        if self.unlabelled and self.unlabelled in request:
            request[self.unlabelled].roi = \
                request[self.unlabelled].roi.union(context_roi)

        logger.debug("upstream %s request: "%self.labels + str(labels_roi))

    def process(self, batch, request):

        affinities_roi = request[self.affinities].roi

        logger.debug("computing ground-truth affinities from labels")
        affinities = malis.seg_to_affgraph(
                batch.arrays[self.labels].data.astype(np.int32),
                self.affinity_neighborhood
        ).astype(np.float32)


        # crop affinities to requested ROI
        offset = affinities_roi.get_offset()
        shift = -offset - self.padding_neg
        crop_roi = affinities_roi.shift(shift)
        crop_roi /= self.spec[self.labels].voxel_size
        crop = crop_roi.get_bounding_box()

        logger.debug("cropping with " + str(crop))
        affinities = affinities[(slice(None),)+crop]

        spec = self.spec[self.affinities].copy()
        spec.roi = affinities_roi
        batch.arrays[self.affinities] = Array(affinities, spec)

        if self.affinities_mask and self.affinities_mask in request:

            if self.labels_mask:

                logger.debug("computing ground-truth affinities mask from "
                             "labels mask")
                affinities_mask = malis.seg_to_affgraph(
                    batch.arrays[self.labels_mask].data.astype(np.int32),
                    self.affinity_neighborhood)
                affinities_mask = affinities_mask[(slice(None),)+crop]

            else:

                affinities_mask = np.ones_like(affinities)

            if self.unlabelled:

                # 1 for all affinities between unlabelled voxels
                unlabelled = (1 - batch.arrays[self.unlabelled].data)
                unlabelled_mask = malis.seg_to_affgraph(
                    unlabelled.astype(np.int32),
                    self.affinity_neighborhood)
                unlabelled_mask = unlabelled_mask[(slice(None),)+crop]

                # 0 for all affinities between unlabelled voxels
                unlabelled_mask = (1 - unlabelled_mask)

                # combine with mask
                affinities_mask = affinities_mask*unlabelled_mask

            affinities_mask = affinities_mask.astype(np.float32)
            batch.arrays[self.affinities_mask] = Array(affinities_mask, spec)

        else:

            if self.labels_mask is not None:
                logger.warning("GT labels does have a mask, but affinities "
                               "mask is not requested.")

        # crop labels to original label ROI
        if self.labels in request:
            roi = request[self.labels].roi
            batch.arrays[self.labels] = batch.arrays[self.labels].crop(roi)

        # same for label mask
        if self.labels_mask and self.labels_mask in request:
            roi = request[self.labels_mask].roi
            batch.arrays[self.labels_mask] = \
                batch.arrays[self.labels_mask].crop(roi)

        # and unlabelled mask
        if self.unlabelled and self.unlabelled in request:
            roi = request[self.unlabelled].roi
            batch.arrays[self.unlabelled] = \
                batch.arrays[self.unlabelled].crop(roi)

        batch.affinity_neighborhood = self.affinity_neighborhood
