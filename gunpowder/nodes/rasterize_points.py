import copy
import logging
import numpy as np
from scipy import ndimage

from .batch_filter import BatchFilter
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from gunpowder.coordinate import Coordinate
from gunpowder.morphology import enlarge_binary_map
from gunpowder.points import PointsKeys, RasterizationSetting

logger = logging.getLogger(__name__)

class RasterizePoints(BatchFilter):
    '''Draw points into a binary array as balls of a given radius.

    Args:

        points (:class:``PointsKeys``):
            The key of the points to rasterize.

        array (:class:``ArrayKey``):
            The key of the binary array to create.

        rastersettings (:class:``RasterizationSetting``, optional):
            Which settings to use to rasterize the points.
    '''

    def __init__(self, points, array, rastersettings=None):

        self.points = points
        self.array = array
        if rastersettings is None:
            self.rastersettings = RasterizationSetting()
        else:
            self.rastersettings = rastersettings
        self.voxel_size = None

    def setup(self):

        dims = self.spec[self.points].roi.dims()
        if self.rastersettings.voxel_size is None:
            self.voxel_size = Coordinate((1,)*dims)
        else:
            assert len(self.rastersettings.voxel_size) == dims, (
                "Given voxel size in raster settings does not match "
                "dimensions of provided points.")
            self.voxel_size = self.rastersettings.voxel_size

        self.provides(
            self.array,
            ArraySpec(
                roi=self.spec[self.points].roi.copy(),
                voxel_size=self.voxel_size))
        self.enable_autoskip()

    def prepare(self, request):

        # TODO: add points request here
        # TODO: optionally add stay_inside_arraytype to request
        pass

    def process(self, batch, request):

        binary_map = self.__get_binary_map(
            batch,
            request,
            self.points,
            self.array)
        spec = self.specs[self.array].copy()
        spec.roi = request[self.array].copy()
        batch.arrays[self.array] = Array(
            data=binary_map,
            spec=spec)

    def __get_binary_map(self, batch, request, points_key, array_key):
        """ requires given point locations to lie within to current bounding box already, because offset of batch is wrong"""

        points = batch.points[points_key]

        voxel_size = self.voxel_size
        shape_bm_array = request.arrays[array_key].get_shape()/voxel_size
        offset_bm_phys = request.arrays[array_key].get_offset()
        binary_map = np.zeros(shape_bm_array, dtype='uint8')


        if self.rastersetting.stay_inside_arraytype is not None:
            mask = batch.arrays[self.rastersetting.stay_inside_arraytype].data
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
                if request.arrays[array_key].contains(Coordinate(batch.points[points_key].data[loc_id].location)):
                    shifted_loc = batch.points[points_key].data[loc_id].location - np.asarray(offset_bm_phys)
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
                binary_map = enlarge_binary_map(binary_map,
                        marker_size_voxel=self.rastersetting.marker_size_voxel,
                       marker_size_physical=self.rastersetting.marker_size_physical,
                       voxel_size=batch.points[points_key].resolution,
                                                donut_inner_radius=self.rastersetting.donut_inner_radius)
                binary_map[mask != object_id] = 0
                binary_map_total += binary_map
                binary_map.fill(0)
            binary_map_total[binary_map_total != 0] = 1
        else:
            for loc_id in points.data.keys():
                if request.arrays[array_key].contains(Coordinate(batch.points[points_key].data[loc_id].location)):
                    shifted_loc = batch.points[points_key].data[loc_id].location - np.asarray(offset_bm_phys)
                    shifted_loc = shifted_loc.astype(np.int32)/voxel_size
                    binary_map[[[loc] for loc in shifted_loc]] = 1
            binary_map_total = enlarge_binary_map(binary_map,
                    marker_size_voxel=self.rastersetting.marker_size_voxel,
                       marker_size_physical=self.rastersetting.marker_size_physical,
                       voxel_size=batch.points[points_key].resolution,
                                                donut_inner_radius=self.rastersetting.donut_inner_radius)
        if len(points.data.keys()) == 0:
            assert np.all(binary_map_total == 0)
        if self.rastersetting.invert_map:
            binary_map_total = np.logical_not(binary_map_total).astype(np.uint8)
        return binary_map_total



