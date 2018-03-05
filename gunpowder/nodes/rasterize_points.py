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
from gunpowder.ndarray import replace
from gunpowder.points import PointsKeys
from gunpowder.points_spec import PointsSpec
from gunpowder.roi import Roi

logger = logging.getLogger(__name__)

class RasterizationSettings(Freezable):
    '''Data structure to store parameters for rasterization of points.

    Args:

        ball_radius (int):

            The radius in world units.

        mask (:class:``ArrayKey``, optional):

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
            mask=None,
            sphere_inner_radius=None,
            fg_value=1,
            bg_value=0):

        if sphere_inner_radius is not None:
            assert ball_radius < sphere_inner_radius, (
                "trying to create a sphere in which the inner radius is larger "
                "or equal than the ball radius")
        self.ball_radius = ball_radius
        self.mask = mask
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

        # request points in a larger area to get rasterization from outside
        # points
        points_roi = request[self.array].roi.grow(
                Coordinate((self.raster_settings.ball_radius,)*self.array_spec.roi.dims()),
                Coordinate((self.raster_settings.ball_radius,)*self.array_spec.roi.dims()))

        # however, restrict the request to the points actually provided
        points_roi = points_roi.intersect(self.spec[self.points].roi)

        request[self.points] = PointsSpec(roi=points_roi)

        if self.raster_settings.mask is not None:
            request[self.raster_settings.mask] = ArraySpec(roi=points_roi)

    def process(self, batch, request):

        points = batch.points[self.points]
        mask = self.raster_settings.mask
        voxel_size = self.spec[self.array].voxel_size

        # get the output array shape
        offset = points.spec.roi.get_begin()/voxel_size
        shape = -(-points.spec.roi.get_shape()/voxel_size) # ceil division
        data_roi = Roi(offset, shape)

        if mask is not None:

            # get all component labels in the mask
            labels = list(np.unique(batch.arrays[mask].data))

            # zero label should be ignored
            if 0 in labels:
                labels.remove(0)

            # create data for the whole points ROI, "or"ed together over
            # individual object masks
            rasterized_points_data = reduce(
                np.logical_or,
                [
                    self.__rasterize(
                        points,
                        data_roi,
                        voxel_size,
                        self.spec[self.array].dtype,
                        self.raster_settings,
                        Array(data=mask.data==label, spec=mask.spec))

                    for label in labels
                ])

        else:

            # create data for the whole points ROI without mask
            rasterized_points_data = self.__rasterize(
                points,
                data_roi,
                voxel_size,
                self.spec[self.array].dtype,
                self.raster_settings)

        # fix bg/fg labelling if requested
        if (self.raster_settings.bg_value != 0 or
            self.raster_settings.fg_value != 1):

            replaced = replace(
                rasterized_points_data,
                [0, 1],
                [self.raster_settings.bg_value, self.raster_settings.fg_value])
            rasterized_points_data = replaced.astype(self.spec[self.array].dtype)

        # create array and crop it to requested roi
        spec = self.spec[self.array].copy()
        spec.roi = data_roi*voxel_size
        rasterized_points = Array(
            data=rasterized_points_data,
            spec=spec)
        batch.arrays[self.array] = rasterized_points.crop(request[self.array].roi)

        # restore requested ROI of points
        if self.points in request:
            request_roi = request[self.points].roi
            points.spec.roi = request_roi
            for i, p in points.data.items():
                if not request_roi.contains(p.location):
                    del points.data[i]

    def __rasterize(self, points, data_roi, voxel_size, dtype, settings, mask_array=None):
        '''Rasterize 'points' into an array with the given 'voxel_size'. If a
        mask array is given, it needs to have the same ROI as the points.'''

        assert mask_array is None or mask_array.spec.roi == points.spec.roi
        assert mask_array is None or mmask_array.spec.voxel_size == voxel_size
        mask = mask_array.data if mask_array is not None else None

        logger.debug("Rasterizing points in %s", points.spec.roi)

        # prepare output array
        rasterized_points = np.zeros(data_roi.get_shape(), dtype=dtype)

        # mark each point with a single voxel
        for point in points.data.values():

            # get the voxel coordinate, 'Coordinate' ensures integer
            v = Coordinate(point.location/voxel_size)

            # get the voxel coordinate relative to output array start
            v -= data_roi.get_begin()

            # skip points outside of mask
            if mask is not None and not mask[v]:
                continue

            logger.debug("Rasterizing point %s at %s",(
                point.location,
                point.location/voxel_size - data_roi.get_begin()))

            # mark the point
            rasterized_points[v] = 1

        # grow points
        enlarge_binary_map(
            rasterized_points,
            settings.ball_radius,
            voxel_size,
            settings.sphere_inner_radius,
            in_place=True)

        if mask_array is not None:
            rasterized_points &= mask

        return rasterized_points
