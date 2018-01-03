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

logger = logging.getLogger(__name__)

class RasterizationSetting(Freezable):
    '''Data structure to store parameters for rasterization of points.

    Args:

        ball_radius_voxel (int):

            Parameter only used, when ``ball_radius_physical`` is not set/set
            to None. Specifies the ball radius in voxel units.

        ball_radius_physical (int):

            If set, overwrites the ``ball_radius_voxel`` parameter. Provides
            the radius in world units. For instance, if ``voxel_size`` is [20,
            10, 10], an ``ball_radius_physical`` of 10 would create a ball with
            a radius of 1 in the x,y-directions and 0 in the z-direction.

        stay_inside_array (:class:``ArrayKey``):

            Used to mask out created balls. The array is assumed to contain
            discrete labels. The object id at the specific point being
            rasterized is used to crop the ball. Blob regions that are located
            outside of the object are masked out, such that the ball is only
            inside the specific object.

        sphere_inner_radius (int):

            If set, instead of a ball, a hollow sphere is rastered. The radius of
            the whole sphere corresponds to the radius specified with
            ``ball_radius_physical`` or ``ball_radius_voxel``. This parameter
            sets the radius of the hollow area and has to be provided in the
            same unit as the specified ball radius.

        voxel_size (:class:``Coordinate``, optional):

            The voxel size of the array to create in world units.
    '''
    def __init__(
            self,
            ball_radius_voxel=1,
            ball_radius_physical=None,
            stay_inside_array=None,
            sphere_inner_radius=None,
            voxel_size=None,
            invert_map=False):

        if sphere_inner_radius is not None:
            if ball_radius_physical is not None:
                ball_radius_check = ball_radius_physical
            else:
                ball_radius_check = ball_radius_voxel
            assert sphere_inner_radius < ball_radius_check, (
                "trying to create a sphere in which the inner radius is larger "
                "than the sphere size")
        self.ball_radius_voxel = ball_radius_voxel
        self.ball_radius_physical = ball_radius_physical
        self.stay_inside_array = stay_inside_array
        self.sphere_inner_radius = sphere_inner_radius
        self.voxel_size = voxel_size
        self.invert_map = invert_map
        self.freeze()

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
        # TODO: optionally add stay_inside_array to request
        pass

    def process(self, batch, request):

        binary_map = self.__get_binary_map(
            batch,
            request,
            self.points,
            self.array)
        spec = self.spec[self.array].copy()
        spec.roi = request[self.array].roi.copy()
        batch.arrays[self.array] = Array(
            data=binary_map,
            spec=spec)

    def __get_binary_map(self, batch, request, points_key, array_key):
        """ requires given point locations to lie within to current bounding box already, because offset of batch is wrong"""

        points = batch.points[points_key]

        logger.debug("Rasterizing %d points...", len(points.data))

        voxel_size = self.voxel_size
        shape_bm_array = request[array_key].roi.get_shape()/voxel_size
        offset_bm_phys = request[array_key].roi.get_offset()
        binary_map = np.zeros(shape_bm_array, dtype='uint8')


        if self.rastersettings.stay_inside_array is not None:
            mask = batch.arrays[self.rastersettings.stay_inside_array].data
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
                    ball_radius_voxel=self.rastersettings.ball_radius_voxel,
                    ball_radius_physical=self.rastersettings.ball_radius_physical,
                    voxel_size=voxel_size,
                    sphere_inner_radius=self.rastersettings.sphere_inner_radius)
                binary_map[mask != object_id] = 0
                binary_map_total += binary_map
                binary_map.fill(0)
            binary_map_total[binary_map_total != 0] = 1
        else:
            for loc_id in points.data.keys():
                if request[array_key].roi.contains(Coordinate(points.data[loc_id].location)):
                    shifted_loc = batch.points[points_key].data[loc_id].location - offset_bm_phys
                    shifted_loc = shifted_loc/voxel_size
                    binary_map[[[loc] for loc in shifted_loc]] = 1
            binary_map_total = enlarge_binary_map(
                binary_map,
                ball_radius_voxel=self.rastersettings.ball_radius_voxel,
                ball_radius_physical=self.rastersettings.ball_radius_physical,
                voxel_size=voxel_size,
                sphere_inner_radius=self.rastersettings.sphere_inner_radius)
        if len(points.data.keys()) == 0:
            assert np.all(binary_map_total == 0)
        if self.rastersettings.invert_map:
            binary_map_total = np.logical_not(binary_map_total).astype(np.uint8)
        return binary_map_total
