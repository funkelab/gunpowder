import copy
import logging
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from .batch_filter import BatchFilter
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from gunpowder.coordinate import Coordinate
from gunpowder.freezable import Freezable
from gunpowder.morphology import enlarge_binary_map, create_ball_kernel
from gunpowder.ndarray import replace
from gunpowder.points import PointsKeys
from gunpowder.points_spec import PointsSpec
from gunpowder.roi import Roi

logger = logging.getLogger(__name__)

class RasterizationSettings(Freezable):
    '''Data structure to store parameters for rasterization of points.

    Args:

        radius (``float`` or ``tuple`` of ``float``):

            The radius (for balls) or sigma (for peaks) in world units.

        mode (``string``):

            One of ``ball`` or ``peak``. If ``ball`` (the default), a ball with the
            given ``radius`` will be drawn. If ``peak``, the point will be
            rasterized as a peak with values :math:`\exp(-|x-p|^2/\sigma)` with
            sigma set by ``radius``.

        mask (:class:`ArrayKey`, optional):

            Used to mask the rasterization of points. The array is assumed to
            contain discrete labels. The object id at the specific point being
            rasterized is used to intersect the rasterization to keep it inside
            the specific object.

        inner_radius_fraction (``float``, optional):

            Only for mode ``ball``.

            If set, instead of a ball, a hollow sphere is rastered. The radius
            of the whole sphere corresponds to the radius specified with
            ``radius``. This parameter sets the radius of the hollow area, as a
            fraction of ``radius``.

        fg_value (``int``, optional):

            Only for mode ``ball``.

            The value to use to rasterize points, defaults to 1.

        bg_value (``int``, optional):

            Only for mode ``ball``.

            The value to use to for the background in the output array,
            defaults to 0.
    '''
    def __init__(
            self,
            radius,
            mode='ball',
            mask=None,
            inner_radius_fraction=None,
            fg_value=1,
            bg_value=0):

        radius = np.array([radius]).flatten()

        if inner_radius_fraction is not None:
            assert (
                inner_radius_fraction > 0.0 and
                inner_radius_fraction < 1.0), (
                    "Inner radius fraction has to be between (excluding) 0 and 1")

        self.radius = radius
        self.mode = mode
        self.mask = mask
        self.inner_radius_fraction = inner_radius_fraction
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

        settings (:class:``RasterizationSettings``, optional):
            Which settings to use to rasterize the points.
    '''

    def __init__(self, points, array, array_spec=None, settings=None):

        self.points = points
        self.array = array
        if array_spec is None:
            self.array_spec = ArraySpec()
        else:
            self.array_spec = array_spec
        if settings is None:
            self.settings = RasterizationSettings(1)
        else:
            self.settings = settings

    def setup(self):

        points_roi = self.spec[self.points].roi

        if self.array_spec.voxel_size is None:
            self.array_spec.voxel_size = Coordinate((1,)*points_roi.dims())

        if self.array_spec.dtype is None:
            if self.settings.mode == 'ball':
                self.array_spec.dtype = np.uint8
            else:
                self.array_spec.dtype = np.float32

        self.array_spec.roi = points_roi.copy()
        self.provides(
            self.array,
            self.array_spec)

        self.enable_autoskip()

    def prepare(self, request):

        if self.settings.mode == 'ball':
            context = np.ceil(self.settings.radius).astype(np.int)
        elif self.settings.mode == 'peak':
            context = np.ceil(2*self.settings.radius).astype(np.int)
        else:
            raise RuntimeError('unknown raster mode %s'%self.settings.mode)

        dims = self.array_spec.roi.dims()
        if len(context) == 1:
            context = context.repeat(dims)

        # request points in a larger area to get rasterization from outside
        # points
        points_roi = request[self.array].roi.grow(
                Coordinate(context),
                Coordinate(context))

        # however, restrict the request to the points actually provided
        points_roi = points_roi.intersect(self.spec[self.points].roi)
        request[self.points] = PointsSpec(roi=points_roi)

        if self.settings.mask is not None:

            mask_voxel_size = self.spec[self.settings.mask].voxel_size
            assert self.spec[self.array].voxel_size == mask_voxel_size, (
                "Voxel size of mask and rasterized volume need to be equal")

            new_mask_roi = points_roi.snap_to_grid(mask_voxel_size)
            if self.settings.mask in request:
                request[self.settings.mask].roi = \
                    request[self.settings.mask].roi.union(new_mask_roi)
            else:
                request[self.settings.mask] = \
                    ArraySpec(roi=new_mask_roi)

    def process(self, batch, request):

        points = batch.points[self.points]
        mask = self.settings.mask
        voxel_size = self.spec[self.array].voxel_size

        # get roi used for creating the new array (points_roi does no
        # necessarily align with voxel size)
        enlarged_vol_roi = points.spec.roi.snap_to_grid(voxel_size)
        offset = enlarged_vol_roi.get_begin() / voxel_size
        shape = enlarged_vol_roi.get_shape() / voxel_size
        data_roi = Roi(offset, shape)

        logger.debug("Points in %s", points.spec.roi)
        for i, point in points.data.items():
            logger.debug("%d, %s", i, point.location)
        logger.debug("Data roi in voxels: %s", data_roi)
        logger.debug("Data roi in world units: %s", data_roi*voxel_size)

        if len(points.data.items()) == 0:
            # If there are no points at all, just create an empty matrix.
            rasterized_points_data = np.zeros(data_roi.get_shape(),
                                              dtype=self.spec[self.array].dtype)
        elif mask is not None:

            mask_array = batch.arrays[mask].crop(enlarged_vol_roi)
            # get those component labels in the mask, that contain points
            labels = []
            for i, point in points.data.items():
                v = Coordinate(point.location / voxel_size)
                v -= data_roi.get_begin()
                labels.append(mask_array.data[v])
            # Make list unique
            labels = list(set(labels))

            # zero label should be ignored
            if 0 in labels:
                labels.remove(0)

            # create data for the whole points ROI, "or"ed together over
            # individual object masks
            rasterized_points_data = np.sum(
                [
                    self.__rasterize(
                        points,
                        data_roi,
                        voxel_size,
                        self.spec[self.array].dtype,
                        self.settings,
                        Array(data=mask_array.data==label, spec=mask_array.spec))

                    for label in labels
                ],
                axis=0)

        else:

            # create data for the whole points ROI without mask
            rasterized_points_data = self.__rasterize(
                points,
                data_roi,
                voxel_size,
                self.spec[self.array].dtype,
                self.settings)

        # fix bg/fg labelling if requested
        if (self.settings.bg_value != 0 or
            self.settings.fg_value != 1):

            replaced = replace(
                rasterized_points_data,
                [0, 1],
                [self.settings.bg_value, self.settings.fg_value])
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

        # restore requested mask
        if mask is not None:
            batch.arrays[mask] = batch.arrays[mask].crop(request[mask].roi)

    def __rasterize(self, points, data_roi, voxel_size, dtype, settings, mask_array=None):
        '''Rasterize 'points' into an array with the given 'voxel_size'''

        mask = mask_array.data if mask_array is not None else None

        logger.debug("Rasterizing points in %s", points.spec.roi)

        # prepare output array
        rasterized_points = np.zeros(data_roi.get_shape(), dtype=dtype)

        # Fast rasterization currently only implemented for mode ball without
        # inner radius set
        use_fast_rasterization = (
            settings.mode == 'ball' and
            settings.inner_radius_fraction is None
        )

        if use_fast_rasterization:

            dims = len(rasterized_points.shape)

            # get structuring element for mode ball
            ball_kernel = create_ball_kernel(settings.radius, voxel_size)
            radius_voxel = Coordinate(np.array(ball_kernel.shape)/2)
            data_roi_base = Roi(
                    offset=Coordinate((0,)*dims),
                    shape=Coordinate(rasterized_points.shape))
            kernel_roi_base = Roi(
                    offset=Coordinate((0,)*dims),
                    shape=Coordinate(ball_kernel.shape))

        # Rasterize volume either with single voxel or with defined struct elememt
        for point in points.data.values():

            # get the voxel coordinate, 'Coordinate' ensures integer
            v = Coordinate(point.location/voxel_size)

            # get the voxel coordinate relative to output array start
            v -= data_roi.get_begin()

            # skip points outside of mask
            if mask is not None and not mask[v]:
                continue

            logger.debug(
                "Rasterizing point %s at %s",
                point.location,
                point.location/voxel_size - data_roi.get_begin())

            if use_fast_rasterization:

                # Calculate where to crop the kernel mask and the rasterized array
                shifted_kernel = kernel_roi_base.shift(v - radius_voxel)
                shifted_data = data_roi_base.shift(-(v - radius_voxel))
                arr_crop = data_roi_base.intersect(shifted_kernel)
                kernel_crop = kernel_roi_base.intersect(shifted_data)
                arr_crop_ind = arr_crop.get_bounding_box()
                kernel_crop_ind = kernel_crop.get_bounding_box()

                rasterized_points[arr_crop_ind] = np.logical_or(
                    ball_kernel[kernel_crop_ind],
                    rasterized_points[arr_crop_ind])

            else:

                rasterized_points[v] = 1

        # grow points
        if not use_fast_rasterization:

            if settings.mode == 'ball':

                enlarge_binary_map(
                    rasterized_points,
                    settings.radius,
                    voxel_size,
                    1.0 - settings.inner_radius_fraction,
                    in_place=True)

            else:

                sigmas = settings.radius/voxel_size

                gaussian_filter(
                    rasterized_points,
                    sigmas,
                    output=rasterized_points,
                    mode='constant')

                # renormalize to have 1 be the highest value
                max_value = np.max(rasterized_points)
                if max_value > 0:
                    rasterized_points /= max_value

        if mask_array is not None:
            # use more efficient bitwise operation when possible
            if settings.mode == 'ball':
                rasterized_points &= mask
            else:
                rasterized_points *= mask

        return rasterized_points
