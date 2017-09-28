import copy
import logging
import numpy as np
import pdb

from gunpowder.volume import Volume, VolumeTypes
from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class AddLongRangeAffinities(BatchFilter):


    def __init__(self, affinity_vectors, volume_type_1=None, volume_type_2=None, affinity_volume_type=None):

        self.volume_type_1 = volume_type_1
        self.volume_type_2 = volume_type_2
        self.affinity_vectors = affinity_vectors

        if volume_type_1 is None:
            self.volume_type_1 = VolumeTypes.PRESYN
        if volume_type_2 is None:
            self.volume_type_2 = VolumeTypes.POSTSYN
        if affinity_volume_type is None:
            self.affinity_volume_type = VolumeTypes.LR_AFFINITIES

        self.skip_next = False



    def setup(self):
        assert self.volume_type_1 in self.spec, "Upstream does not provide %s needed by \
        AddGtAffinities"%self.volume_type_1
        assert self.volume_type_2 in self.spec, "Upstream does not provide %s needed by \
        AddGtAffinities"%self.volume_type_2

        voxel_size = self.spec[self.volume_type_1].voxel_size

        self.upstream_spec = self.get_upstream_provider().spec
        self.upstream_roi = self.upstream_spec.get_total_roi()


        # get maximum offset in each dimension
        self.padding = np.max(np.abs(self.affinity_vectors), axis=0)/voxel_size
        self.padding = tuple(round_to_voxel_size(self.padding, voxel_size))

        logger.debug("padding neg: %s" %np.asarray(self.padding))

        spec = self.spec[self.volume_type_1].copy()
        # if spec.roi is not None:
        #     spec.roi = spec.roi.grow(self.padding, self.padding)

        self.provides(self.affinity_volume_type, spec)

    def prepare(self, request):

        # do nothing if no gt affinities were requested
        if not self.affinity_volume_type in request:
            logger.warn("no affinites requested, will do nothing")
            self.skip_next = True
            return

        del request[self.affinity_volume_type]

        volume_1_roi = request[self.volume_type_1].roi
        logger.debug("downstream %s request: "%self.volume_type_1 + str(volume_1_roi))

        volume_2_roi = request[self.volume_type_2].roi
        logger.debug("downstream %s request: "%self.volume_type_1 + str(volume_2_roi))

        # grow labels ROI to accomodate padding TODO: vol 2
        volume_1_roi = volume_1_roi.grow(self.padding, self.padding)
        volume_2_roi = volume_1_roi.grow(self.padding, self.padding)

        request[self.volume_type_1].roi = volume_1_roi
        request[self.volume_type_2].roi = volume_2_roi

        logger.debug("upstream %s request: "%self.volume_type_1 + str(volume_1_roi))
        logger.debug("upstream %s request: "%self.volume_type_1 + str(volume_2_roi))

        # pdb.set_trace()

    def process(self, batch, request):
        pdb.set_trace()

        # do nothing if no gt affinities were requested
        if self.skip_next:
            self.skip_next = False
            return

        volume_1_roi = request[self.volume_type_1].roi
        volume_2_roi = request[self.volume_type_2].roi

        logger.debug("computing ground-truth affinities from labels")

        # pad batch data
        pdb.set_trace()

        # gt_affinities = malis.seg_to_affgraph(
        #         batch.volumes[self.label_volume_type].data.astype(np.int32),
        #         self.affinity_neighborhood
        # ).astype(np.float32)

        # crop affinities to original label ROI
        # offset = gt_labels_roi.get_offset()
        # shift = -offset - self.padding_neg
        # crop_roi = gt_labels_roi.shift(shift)
        # crop_roi /= self.spec[self.label_volume_type].voxel_size
        # crop = crop_roi.get_bounding_box()

        # logger.debug("cropping with " + str(crop))
        # gt_affinities = gt_affinities[(slice(None),)+crop]

        # spec = self.spec[self.affinity_volume_type].copy()
        # spec.roi = gt_labels_roi
        # batch.volumes[self.affinity_volume_type] = Volume(gt_affinities, spec)

        # crop labels to original label ROI
        # batch.volumes[self.label_volume_type] = batch.volumes[self.label_volume_type].crop(gt_labels_roi)

        # batch.affinity_neighborhood = self.affinity_neighborhood

def round_to_voxel_size(shape, voxel_size):
    voxel_size = np.asarray(voxel_size, dtype=float)
    shape = np.ceil(shape/voxel_size)*voxel_size
    return np.array(shape, dtype='int32')



