import copy
import logging
import numpy as np
from scipy import ndimage

from .batch_filter import BatchFilter
from gunpowder.array import Array
from gunpowder.coordinate import Coordinate
from gunpowder.morphology import enlarge_binary_map
from gunpowder.points import PointsKeys, RasterizationSetting

logger = logging.getLogger(__name__)

class RasterizePoints(BatchFilter):
    ''' Create binary map for points of given PointsKey in batch and add it as array to batch '''
    def __init__(self, arrays, rastersettings=None):
        ''' Add binary map of given PointsKey as array to batch.
        Args:
            arrays (dict, :class:``ArrayKey`` -> :class:``PointsKey``):
                Which arrays to create (keys of the dict) from which points
                (values of the dict).

            rastersettings (dict, :class:``ArrayKey``->:class:``RasterizationSetting``, optional):
                Which settings to use to rasterize the points into arrays.
        '''
        self.arrays = arrays
        if rastersettings is None:
            self.rastersettings = {}
        else:
            self.rastersettings = rastersettings
        self.skip_next = False


    def setup(self):

        self.upstream_spec = self.get_upstream_provider().get_spec()
        self.spec = copy.deepcopy(self.upstream_spec)

        for (array_type, points_type) in self.arrays.items():
            assert points_type in self.spec.points, "Asked for {} from {}, where {} is not provided.".format(array_type, points_type, points_type)
            self.spec.arrays[array_type] = self.spec.points[points_type]

    def get_spec(self):
        return self.spec

    def prepare(self, request):

        self.skip_next = True
        for array_type, points_type in self.arrays.items():
            if array_type in request.arrays:
                assert points_type in request.points
                # if at least one requested array is in self.pointstype_to_arrays, therefore do not skip process
                self.skip_next = False

        if self.skip_next:
            logger.warn("no ArrayKeys of BinaryMask ({}) requested, will do nothing".format(self.arrays.values()))

        if len(self.arrays) == 0:
            self.skip_next = True


    def process(self, batch, request):

        # do nothing if no gt binary maps were requested
        if self.skip_next:
            self.skip_next = False
            return

        for nr, (array_type, points_type) in enumerate(self.arrays.items()):
            if array_type in request.arrays:
                binary_map = self.__get_binary_map(batch, request, points_type, array_type, points=batch.points[points_type])
                batch.arrays[array_type] = Array(data=binary_map,
                                                    roi=request.arrays[array_type])


    def __get_binary_map(self, batch, request, points_type, array_type, points):
        """ requires given point locations to lie within to current bounding box already, because offset of batch is wrong"""

        voxel_size = array_type.voxel_size
        shape_bm_array  = request.arrays[array_type].get_shape()/voxel_size
        offset_bm_phys= request.arrays[array_type].get_offset()
        binary_map       = np.zeros(shape_bm_array, dtype='uint8')

        if array_type in self.rastersettings:
            raster_setting = self.rastersettings[array_type]
        else:
            raster_setting = RasterizationSetting()

        if raster_setting.stay_inside_arraytype is not None:
            mask = batch.arrays[raster_setting.stay_inside_arraytype].data
            if mask.shape>binary_map.shape:
                # assumption: the binary map is centered in the mask array
                offsets = (np.asarray(mask.shape) - np.asarray(binary_map.shape)) / 2.
                slices = [slice(np.floor(offset), np.floor(offset)+bm_shape) for offset, bm_shape in
                          zip(offsets, binary_map.shape)]
                mask = mask[slices]
            assert binary_map.shape == mask.shape, 'shape of newly created rasterized array and shape of mask array ' \
                                                   'as specified with stay_inside_arraytype need to ' \
                                                   'be aligned: %s versus mask shape %s' %(binary_map.shape, mask.shape)
            binary_map_total = np.zeros_like(binary_map)
            object_id_locations = {}
            for loc_id in points.data.keys():
                if request.arrays[array_type].contains(Coordinate(batch.points[points_type].data[loc_id].location)):
                    shifted_loc = batch.points[points_type].data[loc_id].location - np.asarray(offset_bm_phys)
                    shifted_loc = shifted_loc.astype(np.int32)/voxel_size

                    # Get id of this location in the mask
                    object_id = mask[[[loc] for loc in shifted_loc]][0] # 0 index, otherwise numpy array with single number
                    if object_id in object_id_locations:
                        object_id_locations[object_id].append(shifted_loc)
                    else:
                        object_id_locations[object_id] = [shifted_loc]

            # Process all points part of the same object together (for efficiency reason, but also because otherwise if
            # donut flag is set, rasterization would create overlapping rings

            for object_id, location_list in object_id_locations.items():
                for location in location_list:
                    binary_map[[[loc] for loc in location]] = 1
                binary_map = enlarge_binary_map(binary_map, marker_size_voxel=raster_setting.marker_size_voxel,
                       marker_size_physical=raster_setting.marker_size_physical,
                       voxel_size=batch.points[points_type].resolution,
                                                donut_inner_radius=raster_setting.donut_inner_radius)
                binary_map[mask != object_id] = 0
                binary_map_total += binary_map
                binary_map.fill(0)
            binary_map_total[binary_map_total != 0] = 1
        else:
            for loc_id in points.data.keys():
                if request.arrays[array_type].contains(Coordinate(batch.points[points_type].data[loc_id].location)):
                    shifted_loc = batch.points[points_type].data[loc_id].location - np.asarray(offset_bm_phys)
                    shifted_loc = shifted_loc.astype(np.int32)/voxel_size
                    binary_map[[[loc] for loc in shifted_loc]] = 1
            binary_map_total = enlarge_binary_map(binary_map, marker_size_voxel=raster_setting.marker_size_voxel,
                       marker_size_physical=raster_setting.marker_size_physical,
                       voxel_size=batch.points[points_type].resolution,
                                                donut_inner_radius=raster_setting.donut_inner_radius)
        if len(points.data.keys()) == 0:
            assert np.all(binary_map_total == 0)
        if raster_setting.invert_map:
            binary_map_total = np.logical_not(binary_map_total).astype(np.uint8)
        return binary_map_total



