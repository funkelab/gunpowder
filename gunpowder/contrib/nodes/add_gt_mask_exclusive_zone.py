import copy
import logging
import numpy as np
from scipy import ndimage

from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.array import Array, ArrayKeys
from gunpowder.nodes.rasterize_points import RasterizationSettings
from gunpowder.morphology import enlarge_binary_map

logger = logging.getLogger(__name__)

class AddGtMaskExclusiveZone(BatchFilter):
    ''' Create ExclusizeZone mask for a binary map in batch and add it as
    array to batch.

    An ExclusiveZone mask is a bianry mask [0,1] where locations which lie
    within a given distance to the ON (=1) regions (surrounding the ON regions)
    of the given binary map are set to 0, whereas all the others are set to 1.

    Args:

        EZ_masks_to_binary_map(dict, :class:``ArrayKey``->:class:``ArrayKey``):
            Arrays of exclusive zones (keys of dict) to create for which
            binary mask (values of dict).

        gaussian_sigma_for_zone(float, optional): Defines extend of exclusive
            zone around ON region in binary map. Defaults to 1.

        rasterization_setting(:class:``RasterizationSettings``, optional): Which
            rasterization setting to use.
    '''

    def __init__(
            self,
            EZ_masks_to_binary_map,
            gaussian_sigma_for_zone=1,
            rasterization_setting=None):

        self.EZ_masks_to_binary_map = EZ_masks_to_binary_map
        self.gaussian_sigma_for_zone = gaussian_sigma_for_zone
        if rasterization_setting is None:
            self.rasterization_setting = RasterizationSettings()
        else:
            self.rasterization_setting = rasterization_setting
        self.skip_next = False


    def setup(self):

        self.upstream_spec = self.get_upstream_provider().get_spec()
        self.spec = copy.deepcopy(self.upstream_spec)

        for EZ_mask_type, binary_map_type in self.EZ_masks_to_binary_map.items():
            if binary_map_type in self.upstream_spec.arrays:
                self.spec.arrays[EZ_mask_type] = self.spec.arrays[binary_map_type]


    def get_spec(self):
        return self.spec


    def prepare(self, request):

        self.EZ_masks_to_create = []
        for EZ_mask_type, binary_map_type in self.EZ_masks_to_binary_map.items():
            # do nothing if binary mask to create EZ mask around is not requested as well
            if EZ_mask_type in request.arrays:
                # assert that binary mask for which EZ mask is created for is requested
                assert binary_map_type in request.arrays, \
                    "ExclusiveZone Mask for {}, can only be created if {} also requested.".format(EZ_mask_type, binary_map_type)
                # assert that ROI of EZ lies within ROI of binary mask
                assert request.arrays[binary_map_type].contains(request.arrays[EZ_mask_type]),\
                    "EZ mask for {} requested for ROI outside of source's ({}) ROI.".format(EZ_mask_type,binary_map_type)

                self.EZ_masks_to_create.append(EZ_mask_type)
                del request.arrays[EZ_mask_type]

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
            binary_map = batch.arrays[binary_map_type].data
            resolution = batch.arrays[binary_map_type].resolution
            EZ_mask = self.__get_exclusivezone_mask(binary_map, shape_EZ_mask=request.arrays[EZ_mask_type].get_shape(),
                                                    resolution=resolution)

            batch.arrays[EZ_mask_type] = Array(data= EZ_mask,
                                                 roi=request.arrays[EZ_mask_type],
                                                 resolution=resolution)

    def __get_exclusivezone_mask(self, binary_map, shape_EZ_mask, resolution=None):
        ''' Exclusive zone surrounds every synapse. Created by enlarging the ON regions of given binary map
        with different gaussian filter, make it binary and subtract the original binary map from it '''

        shape_diff = np.asarray(binary_map.shape - np.asarray(shape_EZ_mask))
        slices = [slice(diff, shape - diff) for diff, shape in zip(shape_diff, binary_map.shape)]
        relevant_binary_map = binary_map[slices]


        BM_enlarged_binary = enlarge_binary_map(relevant_binary_map,
                                                marker_size_voxel=self.rasterization_setting.marker_size_voxel,
                                                voxel_size=resolution,
                                                marker_size_physical=self.rasterization_setting.marker_size_physical)


        exclusive_zone = np.ones_like(BM_enlarged_binary) - (BM_enlarged_binary - relevant_binary_map)
        return exclusive_zone
