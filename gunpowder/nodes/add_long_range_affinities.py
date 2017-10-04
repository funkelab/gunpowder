import copy
import logging
import numpy as np
import pdb

from gunpowder.volume import Volume, VolumeTypes
from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class AddLongRangeAffinities(BatchFilter):


    def __init__(self, affinity_vectors, volume_type_1=None, volume_type_2=None,
        affinity_volume_type_1=None, affinity_volume_type_2=None):

        self.volume_type_1 = volume_type_1
        self.volume_type_2 = volume_type_2
        self.affinity_volume_type_1 = affinity_volume_type_1
        self.affinity_volume_type_2 = affinity_volume_type_2
        self.affinity_vectors = affinity_vectors

        if volume_type_1 is None:
            self.volume_type_1 = VolumeTypes.PRESYN
        if volume_type_2 is None:
            self.volume_type_2 = VolumeTypes.POSTSYN
        if affinity_volume_type_1 is None:
            self.affinity_volume_type_1 = VolumeTypes.PRE_LR_AFFINITIES
        if affinity_volume_type_2 is None:
            self.affinity_volume_type_2 = VolumeTypes.POST_LR_AFFINITIES

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
        self.padding = np.max(np.abs(self.affinity_vectors), axis=0)
        self.padding = tuple(round_to_voxel_size(self.padding, voxel_size))

        logger.debug("padding neg: %s" %np.asarray(self.padding))

        spec = self.spec[self.volume_type_1].copy()
        # if spec.roi is not None:

        self.provides(self.affinity_volume_type_1, spec)
        self.provides(self.affinity_volume_type_2, spec)

    def prepare(self, request):

        # do nothing if no gt affinities were requested
        if not (self.affinity_volume_type_1 in request and self.affinity_volume_type_2 in request):
            logger.warn("no affinites requested, will do nothing")
            self.skip_next = True
            return

        del request[self.affinity_volume_type_1]
        del request[self.affinity_volume_type_2]

        volume_1_roi = request[self.volume_type_1].roi
        logger.debug("downstream %s request: "%self.volume_type_1 + str(volume_1_roi))

        volume_2_roi = request[self.volume_type_2].roi
        logger.debug("downstream %s request: "%self.volume_type_2 + str(volume_2_roi))

        # grow labels ROI to accomodate padding TODO: vol 2
        volume_1_roi = volume_1_roi.grow(self.padding, self.padding)
        volume_2_roi = volume_2_roi.grow(self.padding, self.padding)

        request[self.volume_type_1].roi = volume_1_roi
        request[self.volume_type_2].roi = volume_2_roi

        logger.debug("upstream %s request: "%self.volume_type_1 + str(volume_1_roi))
        logger.debug("upstream %s request: "%self.volume_type_2 + str(volume_2_roi))

        # pdb.set_trace()

    def process(self, batch, request):

        full_vol1 = batch.volumes[self.volume_type_1]
        full_vol2 = batch.volumes[self.volume_type_2]

        # Both full_vol1 should match
        assert full_vol1.spec.dtype == full_vol2.spec.dtype,\
        "data type of volume 1(%s) and volume 2(%s) should match"%\
        (full_vol1.spec.dtype, full_vol2.spec.dtype)

        assert full_vol1.spec.voxel_size == full_vol2.spec.voxel_size,\
        "data type of volume 1(%s) and volume 2(%s) should match"%\
        (full_vol1.spec.voxel_size,full_vol2.spec.voxel_size)

        # do nothing if no gt affinities were requested
        if self.skip_next:
            self.skip_next = False
            return

        logger.debug("computing ground-truth affinities from labels")

        # Calculate affinities 1: from vol2 onto vol1

        # Initialize affinity map
        request_vol = request[self.affinity_volume_type_1]
        affinity_map = np.zeros(
            (len(self.affinity_vectors),) +
            tuple(request_vol.roi.get_shape()/request_vol.voxel_size), dtype=full_vol1.spec.dtype)

        # calculate affinities
        vol1 = full_vol1.crop(request_vol.roi)
        for i, vector in enumerate(self.affinity_vectors):
            vol2 = full_vol2.crop(request_vol.roi.shift(tuple(-vector)))
            affinity_map[i,:,:,:] = np.bitwise_and(vol1.data, vol2.data)


        batch.volumes[self.affinity_volume_type_1] = Volume(affinity_map,
            spec=request[self.affinity_volume_type_1].copy())

        batch.volumes[self.affinity_volume_type_1].attrs['affinity_vectors'] =\
         self.affinity_vectors

        # Calculate affinities 2: from vol1 onto vol2

        # Initialize affinity map
        request_vol = request[self.affinity_volume_type_2]
        affinity_map = np.zeros(
            (len(self.affinity_vectors),) +
            tuple(request_vol.roi.get_shape()/request_vol.voxel_size), dtype=full_vol1.spec.dtype)

        # calculate affinities
        vol2 = full_vol2.crop(request_vol.roi)
        for i, vector in enumerate(self.affinity_vectors):
            vol1 = full_vol1.crop(request_vol.roi.shift(tuple(vector)))
            affinity_map[i,:,:,:] = np.bitwise_and(vol1.data, vol2.data)


        batch.volumes[self.affinity_volume_type_2] = Volume(affinity_map,
            spec=request[self.affinity_volume_type_2].copy())

        batch.volumes[self.affinity_volume_type_2].attrs['affinity_vectors'] =\
         self.affinity_vectors



        # Crop all other requests
        for volume_type, volume in request.volume_specs.items():
            batch.volumes[volume_type] = batch.volumes[volume_type].crop(volume.roi)

        for points_type, points in request.points_specs.items():
            recropped = batch.points[points_type].spec.roi = points.roi
            batch.points[points_type] = recropped


def round_to_voxel_size(shape, voxel_size):
    voxel_size = np.asarray(voxel_size, dtype=float)
    shape = np.ceil(shape/voxel_size)*voxel_size
    return np.array(shape, dtype='int32')



