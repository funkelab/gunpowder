import copy
import logging
import numpy as np
from scipy import ndimage

from .batch_filter import BatchFilter
from gunpowder.volume import Volume, VolumeType
from gunpowder.points import PointsType

logger = logging.getLogger(__name__)

class AddGtMaskExclusiveZone(BatchFilter):

    def __init__(self):
        self.skip_next = False


    def setup(self):

        self.upstream_spec = self.get_upstream_provider().get_spec()
        self.spec = copy.deepcopy(self.upstream_spec)

        self.EZ_masks_to_binary_map = {VolumeType.GT_MASK_EXCLUSIVEZONE_PRESYN: VolumeType.GT_BM_PRESYN,
                                       VolumeType.GT_MASK_EXCLUSIVEZONE_POSTSYN: VolumeType.GT_BM_POSTSYN}

        for EZ_mask_type, binary_map_type in self.EZ_masks_to_binary_map.items():
            if binary_map_type in self.upstream_spec.volumes:
                self.spec.volumes[EZ_mask_type] = self.spec.volumes[binary_map_type]


    def get_spec(self):
        return self.spec


    def prepare(self, request):

        self.EZ_masks_to_create = []
        for EZ_mask_type, binary_map_type in self.EZ_masks_to_binary_map.items():
            # do nothing if binary mask to create EZ mask around is not requested as well
            if EZ_mask_type in request.volumes:
                # assert that binary mask for which EZ mask is created for is requested
                assert binary_map_type in request.volumes, \
                    "ExclusiveZone Mask for {}, can only be created if {} also requestd.".format(mask_volume_type, binary_mask_volume_type)
                # assert that ROI of EZ lies within ROI of binary mask
                assert request.volumes[binary_map_type].contains(request.volumes[EZ_mask_type]),\
                    "EZ mask for {} requested for ROI outside of source's ({}) ROI.".format(EZ_mask_type,binary_mask_type)

                self.EZ_masks_to_create.append(EZ_mask_type)
                del request.volumes[EZ_mask_type]

        if len(self.EZ_masks_to_create) == 0:
            logger.warn("no ExclusiveZone Masks requested, will do nothing")
            self.skip_next = True


    def process(self, batch, request):

        # do nothing if no gt binary maps were requested
        if self.skip_next:
            self.skip_next = False
            return

        for EZ_mask_type in self.EZ_masks_to_create:
            binary_map_type = self.EZ_masks_to_binary_map[EZ_mask_type]
            binary_map = batch.volumes[binary_map_type].data

            interpolate = {VolumeType.GT_MASK_EXCLUSIVEZONE_PRESYN: True,
                           VolumeType.GT_MASK_EXCLUSIVEZONE_POSTSYN: True}[EZ_mask_type]
            EZ_mask = self.__get_exclusivezone_mask(binary_map, shape_EZ_mask=np.asarray(request.volumes[EZ_mask_type].get_shape()), gaussian_sigma_for_zone=2)
            batch.volumes[EZ_mask_type] = Volume(data= EZ_mask,
                                                 roi=request.volumes[EZ_mask_type],
                                                 resolution=batch.volumes[binary_map_type].resolution,
                                                 interpolate=interpolate)


    def __get_exclusivezone_mask(self, binary_map, shape_EZ_mask, gaussian_sigma_for_zone=2):
        ''' Exclusive zone surround every synapse. Created by subtracting two binary masks with different gaussian
        blur strengths (sigmas)'''

        shape_diff = binary_map.shape - shape_EZ_mask
        relevant_binary_map = binary_map[(shape_diff[0] // 2) : (binary_map.shape[0] - shape_diff[0]//2),
                                         (shape_diff[1] // 2) : (binary_map.shape[1] - shape_diff[1]//2),
                                         (shape_diff[2] // 2) : (binary_map.shape[2] - shape_diff[2]//2)]

        BM_enlarged_cont = ndimage.filters.gaussian_filter(relevant_binary_map, sigma=gaussian_sigma_for_zone)
        BM_enlarged_binary = np.zeros_like(relevant_binary_map)
        BM_enlarged_binary[np.nonzero(BM_enlarged_cont)] = 1

        exclusive_zone = np.ones_like(BM_enlarged_binary) - (BM_enlarged_binary - relevant_binary_map)
        return exclusive_zone
