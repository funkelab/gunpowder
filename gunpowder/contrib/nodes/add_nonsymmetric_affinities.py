import copy
import logging
import numpy as np
import pdb

from gunpowder.array import Array
from gunpowder.nodes.batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class AddNonsymmetricAffinities(BatchFilter):


    def __init__(
            self,
            affinity_vectors,
            array_key_1,
            array_key_2,
            affinity_array_key_1,
            affinity_array_key_2):

        self.array_key_1 = array_key_1
        self.array_key_2 = array_key_2
        self.affinity_array_key_1 = affinity_array_key_1
        self.affinity_array_key_2 = affinity_array_key_2
        self.affinity_vectors = affinity_vectors

    def setup(self):
        assert self.array_key_1 in self.spec, "Upstream does not provide %s needed by \
        AddNonsymmetricAffinities"%self.array_key_1
        assert self.array_key_2 in self.spec, "Upstream does not provide %s needed by \
        AddNonsymmetricAffinities"%self.array_key_2

        voxel_size = self.spec[self.array_key_1].voxel_size

        self.upstream_spec = self.get_upstream_provider().spec
        self.upstream_roi = self.upstream_spec.get_total_roi()


        # get maximum offset in each dimension
        self.padding = np.max(np.abs(self.affinity_vectors), axis=0)
        self.padding = tuple(round_to_voxel_size(self.padding, voxel_size))

        logger.debug("padding neg: %s" %np.asarray(self.padding))

        spec = self.spec[self.array_key_1].copy()
        # if spec.roi is not None:

        self.provides(self.affinity_array_key_1, spec)
        self.provides(self.affinity_array_key_2, spec)
        self.enable_autoskip()

    def prepare(self, request):

        array_1_roi = request[self.array_key_1].roi
        logger.debug("downstream %s request: "%self.array_key_1 + str(array_1_roi))

        array_2_roi = request[self.array_key_2].roi
        logger.debug("downstream %s request: "%self.array_key_2 + str(array_2_roi))

        # grow labels ROI to accomodate padding TODO: vol 2
        array_1_roi = array_1_roi.grow(self.padding, self.padding)
        array_2_roi = array_2_roi.grow(self.padding, self.padding)

        request[self.array_key_1].roi = array_1_roi
        request[self.array_key_2].roi = array_2_roi

        logger.debug("upstream %s request: "%self.array_key_1 + str(array_1_roi))
        logger.debug("upstream %s request: "%self.array_key_2 + str(array_2_roi))

        # pdb.set_trace()

    def process(self, batch, request):

        full_vol1 = batch.arrays[self.array_key_1]
        full_vol2 = batch.arrays[self.array_key_2]

        # Both full_vol1 should match
        assert full_vol1.spec.dtype == full_vol2.spec.dtype,\
        "data type of array 1(%s) and array 2(%s) should match"%\
        (full_vol1.spec.dtype, full_vol2.spec.dtype)

        assert full_vol1.spec.voxel_size == full_vol2.spec.voxel_size,\
        "data type of array 1(%s) and array 2(%s) should match"%\
        (full_vol1.spec.voxel_size,full_vol2.spec.voxel_size)

        logger.debug("computing ground-truth affinities from labels")

        # Calculate affinities 1: from vol2 onto vol1

        # Initialize affinity map
        request_vol = request[self.affinity_array_key_1]
        affinity_map = np.zeros(
            (len(self.affinity_vectors),) +
            tuple(request_vol.roi.get_shape()/request_vol.voxel_size), dtype=full_vol1.spec.dtype)

        # calculate affinities
        vol1 = full_vol1.crop(request_vol.roi)
        for i, vector in enumerate(self.affinity_vectors):
            vol2 = full_vol2.crop(request_vol.roi.shift(tuple(-vector)))
            affinity_map[i,:,:,:] = np.bitwise_and(vol1.data, vol2.data)


        batch.arrays[self.affinity_array_key_1] = Array(affinity_map,
            spec=request[self.affinity_array_key_1].copy())

        batch.arrays[self.affinity_array_key_1].attrs['affinity_vectors'] =\
         self.affinity_vectors

        # Calculate affinities 2: from vol1 onto vol2

        # Initialize affinity map
        request_vol = request[self.affinity_array_key_2]
        affinity_map = np.zeros(
            (len(self.affinity_vectors),) +
            tuple(request_vol.roi.get_shape()/request_vol.voxel_size), dtype=full_vol1.spec.dtype)

        # calculate affinities
        vol2 = full_vol2.crop(request_vol.roi)
        for i, vector in enumerate(self.affinity_vectors):
            vol1 = full_vol1.crop(request_vol.roi.shift(tuple(vector)))
            affinity_map[i,:,:,:] = np.bitwise_and(vol1.data, vol2.data)


        batch.arrays[self.affinity_array_key_2] = Array(affinity_map,
            spec=request[self.affinity_array_key_2].copy())

        batch.arrays[self.affinity_array_key_2].attrs['affinity_vectors'] =\
         self.affinity_vectors



        # Crop all other requests
        for array_key, array in request.array_specs.items():
            batch.arrays[array_key] = batch.arrays[array_key].crop(array.roi)

        for points_key, points in request.points_specs.items():
            recropped = batch.points[points_key].spec.roi = points.roi
            batch.points[points_key] = recropped


def round_to_voxel_size(shape, voxel_size):
    voxel_size = np.asarray(voxel_size, dtype=float)
    shape = np.ceil(shape/voxel_size)*voxel_size
    return np.array(shape, dtype='int32')
