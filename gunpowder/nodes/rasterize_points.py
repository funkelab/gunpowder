import copy
import logging
import numpy as np
from scipy import ndimage

from .batch_filter import BatchFilter
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from gunpowder.coordinate import Coordinate
from gunpowder.freezable import Freezable
from gunpowder.morphology import enlarge_binary_map
from gunpowder.points import PointsKeys
from gunpowder.points_spec import PointsSpec

logger = logging.getLogger(__name__)

class RasterizationSettings(Freezable):
    '''Data structure to store parameters for rasterization of points.

    Args:

        ball_radius (int):

            The radius in world units.

        stay_inside_array (:class:``ArrayKey``, optional):

            Used to mask out created balls. The array is assumed to contain
            discrete labels. The object id at the specific point being
            rasterized is used to intersect the ball. This keeps the ball
            inside the specific object.

        sphere_inner_radius (int, optional):

            If set, instead of a ball, a hollow sphere is rastered. The radius of
            the whole sphere corresponds to the radius specified with
            ``ball_radius``. This parameter sets the radius of the hollow area.

        fg_value (int, optional):

            The value to use to rasterize points, defaults to 1.

        bg_value (int, optional):

            The value to use to for the background in the output array,
            defaults to 0.
    '''
    def __init__(
            self,
            ball_radius,
            stay_inside_array=None,
            sphere_inner_radius=None,
            fg_value=1,
            bg_value=0):

        if sphere_inner_radius is not None:
            assert ball_radius < sphere_inner_radius, (
                "trying to create a sphere in which the inner radius is larger "
                "or equal than the ball radius")
        self.ball_radius = ball_radius
        self.stay_inside_array = stay_inside_array
        self.sphere_inner_radius = sphere_inner_radius
        self.fg_value = fg_value
        self.bg_value = bg_value
        self.freeze()

class RasterizePoints(BatchFilter):
    '''Draw points into a binary array as balls of a given radius.

    Args:

        points (:class:``PointsKeys``):
            The key of the points to rasterize.

        array (:class:``ArrayKey``):
            The key of the binary array to create.

        array_spec (:class:``ArraySpec``, optional):

            The spec of the array to create. Use this to set the datatype and
            voxel size.

        raster_settings (:class:``RasterizationSettings``, optional):
            Which settings to use to rasterize the points.
    '''

    def __init__(self, points, array, array_spec=None, raster_settings=None):

        self.points = points
        self.array = array
        if array_spec is None:
            self.array_spec = ArraySpec()
        else:
            self.array_spec = array_spec
        if raster_settings is None:
            self.raster_settings = RasterizationSettings(1)
        else:
            self.raster_settings = raster_settings

    def setup(self):

        points_roi = self.spec[self.points].roi

        if self.array_spec.voxel_size is None:
            self.array_spec.voxel_size = Coordinate((1,)*points_roi.dims())

        if self.array_spec.dtype is None:
            self.array_spec.dtype = np.uint8

        self.array_spec.roi = points_roi.copy()
        self.provides(
            self.array,
            self.array_spec)

        self.enable_autoskip()

    def prepare(self, request):

        points_roi = request[self.array].roi.grow(
                Coordinate((self.raster_settings.ball_radius,)*self.array_spec.roi.dims()),
                Coordinate((self.raster_settings.ball_radius,)*self.array_spec.roi.dims()))

        request[self.points] = PointsSpec(roi=points_roi)

        # TODO: add stay_inside_array to request if used

    def process(self, batch, request):

        binary_map = self.__get_binary_map(
            batch,
            request,
            self.points,
            self.array).astype(self.array_spec.dtype)

        spec = self.spec[self.array].copy()
        spec.roi = request[self.array].roi.copy()

        batch.arrays[self.array] = Array(
            data=binary_map,
            spec=spec)

    def __get_binary_map(self, batch, request, points_key, array_key):
        """ requires given point locations to lie within to current bounding box already, because offset of batch is wrong"""

        points = batch.points[points_key]

        logger.debug("Rasterizing %d points...", len(points.data))

        voxel_size = self.array_spec.voxel_size
        dtype = self.array_spec.dtype

        shape_bm_array = request[array_key].roi.get_shape()/voxel_size
        offset_bm_phys = request[array_key].roi.get_offset()
        binary_map = np.zeros(shape_bm_array, dtype=dtype)

        if self.raster_settings.stay_inside_array is not None:
            mask = batch.arrays[self.raster_settings.stay_inside_array].data
            if mask.shape>binary_map.shape:
                # assumption: the binary map is centered in the mask array
                offsets = (np.asarray(mask.shape) - np.asarray(binary_map.shape)) / 2.
                slices = [slice(np.floor(offset), np.floor(offset)+bm_shape) for offset, bm_shape in
                          zip(offsets, binary_map.shape)]
                mask = mask[slices]
            assert binary_map.shape == mask.shape, 'shape of newly created rasterized array and shape of mask array ' \
                                                   'as specified with stay_inside_array need to ' \
                                                   'be aligned: %s versus mask shape %s' %(binary_map.shape, mask.shape)
            binary_map_total = np.zeros_like(binary_map)
            object_id_locations = {}
            for loc_id in points.data.keys():
                if request[array_key].roi.contains(Coordinate(batch.points[points_key].data[loc_id].location)):
                    shifted_loc = batch.points[points_key].data[loc_id].location - offset_bm_phys
                    shifted_loc = shifted_loc/voxel_size

                    # Get id of this location in the mask
                    object_id = mask[[[loc] for loc in shifted_loc]][0] # 0 index, otherwise numpy array with single number
                    if object_id in object_id_locations:
                        object_id_locations[object_id].append(shifted_loc)
                    else:
                        object_id_locations[object_id] = [shifted_loc]

            # Process all points part of the same object together (for efficiency reason, but also because otherwise if
            # sphere flag is set, rasterization would create overlapping rings

            for object_id, location_list in object_id_locations.items():
                for location in location_list:
                    binary_map[[[loc] for loc in location]] = 1
                binary_map = enlarge_binary_map(
                    binary_map,
                    radius=self.raster_settings.ball_radius,
                    voxel_size=voxel_size,
                    ring_inner=self.raster_settings.sphere_inner_radius)
                binary_map_total[mask == object_id] = binary_map[mask == object_id]
                binary_map[:] = 0
        else:
            for loc_id in points.data.keys():
                if request[array_key].roi.contains(Coordinate(points.data[loc_id].location)):
                    shifted_loc = batch.points[points_key].data[loc_id].location - offset_bm_phys
                    shifted_loc = shifted_loc/voxel_size
                    binary_map[[[loc] for loc in shifted_loc]] = 1
            binary_map_total = enlarge_binary_map(
                binary_map,
                radius=self.raster_settings.ball_radius,
                voxel_size=voxel_size,
                ring_inner=self.raster_settings.sphere_inner_radius)
        if len(points.data.keys()) == 0:
            assert np.all(binary_map_total == 0)

        if (self.raster_settings.fg_value is not 1 or
            self.raster_settings.bg_value is not 0):

            old_values = np.array([0, 1])
            new_values = np.array([
                self.raster_settings.bg_value,
                self.raster_settings.fg_value])

            indices = np.digitize(
                binary_map_total.ravel(),
                old_values,
                right=True)
            binary_map_total = new_values[indices].reshape(binary_map_total.shape)

        return binary_map_total
