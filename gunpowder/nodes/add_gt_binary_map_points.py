import copy
import logging
import numpy as np
from scipy import ndimage

from .batch_filter import BatchFilter
from gunpowder.coordinate import Coordinate
from gunpowder.volume import Volume, VolumeTypes
from gunpowder.points import PointsTypes, RasterizationSetting, enlarge_binary_map


logger = logging.getLogger(__name__)

class AddGtBinaryMapOfPoints(BatchFilter):
    ''' Create binary map for points of given PointsType in batch and add it as volume to batch '''
    def __init__(self, pointstype_to_volumetypes, pointstype_to_rastersettings=None):
        ''' Add binary map of given PointsType as volume to batch.
        Args:
           pointstype_to_volumetypes: dict, e.g. {PointsType.PRESYN: VolumeTypes.GT_BM_PRESYN} creates a binary map
                                      of points in PointsType.PRESYN and adds the created binary map
                                      as a volume of type VolumeTypes.GT_BM_PRESYN to the batch if requested.
           pointstype_to_rastersettings: dict. indicating which kind of rasterization to use for specific pointstype.
                                    Dict maps PointsType to instance of points.RasterizationSetting
        '''
        self.pointstype_to_volumetypes = pointstype_to_volumetypes
        if pointstype_to_rastersettings is None:
            self.pointstype_to_rastersettings = {}
        else:
            self.pointstype_to_rastersettings = pointstype_to_rastersettings
        self.skip_next = False


    def setup(self):

        self.upstream_spec = self.get_upstream_provider().get_spec()
        self.spec = copy.deepcopy(self.upstream_spec)

        for (points_type, volume_type) in self.pointstype_to_volumetypes.items():
            assert points_type in self.spec.points, "Asked for {} from {}, where {} is not provided.".format(volume_type, points_type, points_type)
            self.spec.volumes[volume_type] = self.spec.points[points_type]


    def get_spec(self):
        return self.spec


    def prepare(self, request):

        self.skip_next = True
        for points_type, volume_type in self.pointstype_to_volumetypes.items():
            if volume_type in request.volumes:
                del request.volumes[volume_type]
                assert points_type in request.points
                # if at least one requested volume is in self.pointstype_to_volumes, therefore do not skip process
                self.skip_next = False

        if self.skip_next:
            logger.warn("no VolumeTypes of BinaryMask ({}) requested, will do nothing".format(self.pointstype_to_volumetypes.values()))

        if len(self.pointstype_to_volumetypes) == 0:
            self.skip_next = True


    def process(self, batch, request):

        # do nothing if no gt binary maps were requested
        if self.skip_next:
            self.skip_next = False
            return

        for nr, (points_type, volume_type) in enumerate(self.pointstype_to_volumetypes.items()):
            if volume_type in request.volumes:
                binary_map = self.__get_binary_map(batch, request, points_type, volume_type, points=batch.points[points_type])
                resolution = batch.points[points_type].resolution
                batch.volumes[volume_type] = Volume(data=binary_map,
                                                    roi = request.volumes[volume_type],
                                                    resolution = resolution)


    def __get_binary_map(self, batch, request, points_type, volume_type, points):
        """ requires given point locations to lie within to current bounding box already, because offset of batch is wrong"""

        shape_bm_volume  = request.volumes[volume_type].get_shape()
        offset_bm_volume = request.volumes[volume_type].get_offset()
        binary_map       = np.zeros(shape_bm_volume, dtype='uint8')

        if points_type in self.pointstype_to_rastersettings:
            raster_setting = self.pointstype_to_rastersettings[points_type]
        else:
            raster_setting = RasterizationSetting()
        marker = raster_setting.marker_type
        for loc_id in points.data.keys():
            # check if location lies inside bounding box
            if request.volumes[volume_type].contains(Coordinate(batch.points[points_type].data[loc_id].location)):
                shifted_loc = batch.points[points_type].data[loc_id].location - np.asarray(offset_bm_volume)
                shifted_loc = shifted_loc.astype(np.int32)
                if marker == 'blob':
                    binary_map[[[loc] for loc in shifted_loc]] = 1
                elif marker == 'gaussian':
                    marker_size = 1
                    marker_locs = tuple( slice( max(0, shifted_loc[dim] - marker_size),
                                                min(shape_bm_volume[dim]-1, shifted_loc[dim] + marker_size))
                                                for dim in range(len(shape_bm_volume)))
                    # set to 255 to keep binary map as uint8. That is beneficial to get 'roundish' blob around locations
                    # as smallest values which are produced by gaussian filtering are then set to zero instead of a very small float
                    # resulting in a binary map which is OFF at those locations instead of ON.
                    binary_map[marker_locs] = 255


        # return mask where location is marked as a blob. Set marker_size_voxel to 0, if you require a single point.
        if marker == 'blob':
            binary_map = enlarge_binary_map(binary_map, marker_size_voxel=raster_setting.marker_size_voxel,
                               marker_size_physical=raster_setting.marker_size_physical,
                               voxel_size=batch.points[points_type].resolution)
            return binary_map

        # return mask where location is marked as a gaussian 'blob', produces more square like blobs and does not
        # take into account anisotropy of the data.
        elif marker == 'gaussian':
            binary_map = ndimage.filters.gaussian_filter(binary_map, sigma=1.)
            binary_map_gaussian = np.zeros_like(binary_map, dtype='uint8')
            binary_map_gaussian[np.nonzero(binary_map)] = 1
            return binary_map_gaussian

