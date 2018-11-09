from __future__ import print_function, division
import logging
import numpy as np
import random
from gunpowder.roi import Roi
from gunpowder.coordinate import Coordinate
from gunpowder.points import Point

from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)


class Jitter(BatchFilter):
    def __init__(
            self,
            prob_slip=0,
            prob_shift=0,
            sigma=0,
            jitter_axis=0):

        self.prob_slip = prob_slip
        self.prob_shift = prob_shift
        self.sigma = sigma
        self.jitter_axis = jitter_axis

        self.ndim = None
        self.jitter_sigmas = None
        self.shift_array = None
        self.lcm_voxel_size = None

    def prepare(self, request):
        self.ndim = request.get_total_roi().dims()
        assert self.jitter_axis in range(self.ndim)

        try:
            self.jitter_sigmas = tuple(self.sigma)
        except TypeError:
            self.jitter_sigmas = [float(self.sigma)] * self.ndim
            self.jitter_sigmas[self.jitter_axis] = 0.0
            self.jitter_sigmas = tuple(self.jitter_sigmas)

        assert len(self.jitter_sigmas) == self.ndim
        assert self.jitter_sigmas[self.jitter_axis] == 0.0

        has_nonzero = False
        for sigma in self.jitter_sigmas:
            if sigma != 0.0:
                has_nonzero = True
                break
        assert has_nonzero

        if not request.array_specs:
            raise ValueError("Request passed to Jitter node must contain at least one array key. " +
                             "Check to make sure that Jitter node is not upstream of a RandomLocation node " +
                             "with an ensure_nonempty argument.")

        self.lcm_voxel_size = self.spec.get_lcm_voxel_size(array_keys=request.array_specs.keys())
        assert self.lcm_voxel_size

        roi_shape = request.get_total_roi().get_shape()
        assert roi_shape // self.lcm_voxel_size * self.lcm_voxel_size == roi_shape, \
            "total roi shape {} must be divisible by least common voxel size {}".format(roi_shape, self.lcm_voxel_size)
        roi_shape_adjusted = roi_shape // self.lcm_voxel_size
        jitter_axis_len = roi_shape_adjusted[self.jitter_axis]

        self.shift_array = self.construct_global_shift_array(jitter_axis_len,
                                                             self.jitter_sigmas,
                                                             self.prob_slip,
                                                             self.prob_shift,
                                                             self.lcm_voxel_size)

        for key, spec in request.items():
            sub_shift_array = self.get_sub_shift_array(request.get_total_roi(), spec.roi,
                                                       self.shift_array, self.jitter_axis, self.lcm_voxel_size)
            updated_roi = self.compute_upstream_roi(spec.roi, sub_shift_array)
            spec.roi.set_offset(updated_roi.get_offset())
            spec.roi.set_shape(updated_roi.get_shape())
            request[key] = spec

    def process(self, batch, request):
        for array_key, array in batch.arrays.items():
            sub_shift_array = self.get_sub_shift_array(request.get_total_roi(), array.spec.roi,
                                                       self.shift_array, self.jitter_axis, self.lcm_voxel_size)
            array.data = self.shift_and_crop(array.data,
                                             request[array_key].roi.get_shape(),
                                             sub_shift_array,
                                             array.spec.voxel_size)
            array.spec.roi = request[array_key].roi
            assert request[array_key].roi.get_shape() == Coordinate(array.data.shape) * self.lcm_voxel_size, \
                'request roi shape {} is not the same as generated array shape {}'.format(
                    request[array_key].roi.get_shape(), array.data.shape)
            batch[array_key] = array

        for points_key, points in batch.points.items():
            sub_shift_array = self.get_sub_shift_array(request.get_total_roi(), points.spec.roi,
                                                       self.shift_array, self.jitter_axis, self.lcm_voxel_size)
            points = self.shift_points(points,
                                       request[points_key].roi,
                                       sub_shift_array,
                                       self.jitter_axis,
                                       self.lcm_voxel_size)
            batch[points_key] = points

    def shift_and_crop(self, arr, roi_shape, sub_shift_array, voxel_size):
        """ Shift an array received from upstream and crop it to the target downstream region

        :param arr: an array of upstream data to be jittered and cropped
        :param roi_shape: the shape of the downstream ROI
        :param sub_shift_array: the cropped section of the global shift array that applies to this specific request
        :param voxel_size: the voxel sizes of the data in the array
        :return: an array of shape roi_shape that contains the array to be passed downstream
        """

        array_jitter_axis_len = arr.shape[self.jitter_axis]
        sub_shift_array_len = len(sub_shift_array)
        assert array_jitter_axis_len % sub_shift_array_len == 0, \
            "array jitter axis length {} is not divisible by the sub_shift_array length {}".format(
                arr.shape[self.jitter_axis], sub_shift_array.shape[0])

        voxel_ratio = array_jitter_axis_len // sub_shift_array_len

        # assumption: each sub shift array element divides evenly by the voxel size
        rescaled_sub_shift_array = sub_shift_array // np.array(voxel_size, dtype=int)

        max_shift = rescaled_sub_shift_array.max(axis=0)
        batch = arr.copy()
        batch_view = np.moveaxis(batch, self.jitter_axis, 0)
        for index, plane in enumerate(batch_view):
            adjusted_index = index // voxel_ratio
            shift = rescaled_sub_shift_array[adjusted_index, :] - max_shift
            shift = np.delete(shift, self.jitter_axis, axis=0)
            assert(len(shift) == plane.ndim)
            plane = np.roll(plane, shift, axis=tuple(range(len(shift))))
            batch_view[index] = plane

        adjusted_roi_shape = Coordinate(roi_shape) // Coordinate(voxel_size)

        sl = tuple(slice(0, adjusted_roi_shape[index]) for index in range(self.ndim))
        return batch[sl]

    @staticmethod
    def shift_points(points, request_roi, sub_shift_array, jitter_axis, lcm_voxel_size):
        """ Shift a set of points received from upstream and crop out those not the the target downstream region

        :param points: the points from upstream
        :param request_roi: the downstream ROI
        :param sub_shift_array: the cropped section of the global shift array that applies to this specific request
        :param jitter_axis: the axis to perform the jitter along
        :param lcm_voxel_size: the least common voxel size for the arrays in the request
        :return a Points object with the updated point locations and ROI
        """

        data = points.data
        spec = points.spec
        jitter_axis_start_pos = spec.roi.get_offset()[jitter_axis]

        shifted_data = {}
        for id_, point in data.items():
            loc = Coordinate(point.location)
            jitter_axis_position = loc[jitter_axis]
            shift_array_index = (jitter_axis_position - jitter_axis_start_pos) // lcm_voxel_size[jitter_axis]
            assert(shift_array_index >= 0)
            shift = Coordinate(sub_shift_array[shift_array_index])
            new_loc = loc + shift
            if request_roi.contains(new_loc):
                shifted_data[id_] = Point(new_loc)

        points.data = shifted_data
        points.spec.roi = request_roi
        return points

    @staticmethod
    def get_sub_shift_array(total_roi, item_roi, shift_array, jitter_axis, lcm_voxel_size):
        """ Slices the global shift array to return the sub-shift array to shift an item in the request

        :param total_roi: the total roi of the request
        :param item_roi: the roi of the item (array or points) being shifted
        :param shift_array: the shift array for the total_roi
        :param jitter_axis: the axis along which we are jittering
        :param lcm_voxel_size: the least common voxel size for the arrays in the request
        :return: the portion of the global shift array that should be used to shift the item
        """
        item_offset_from_total = item_roi.get_offset() - total_roi.get_offset()
        offset_in_jitter_axis = item_offset_from_total[jitter_axis] // lcm_voxel_size[jitter_axis]
        len_in_jitter_axis = item_roi.get_shape()[jitter_axis] // lcm_voxel_size[jitter_axis]
        return shift_array[offset_in_jitter_axis: offset_in_jitter_axis + len_in_jitter_axis]

    @staticmethod
    def construct_global_shift_array(jitter_axis_len, jitter_sigmas, prob_slip, prob_shift, lcm_voxel_size):
        """ Sets the attribute variable self.shift_array

        :param jitter_axis_len: the length of the jitter axis
        :param jitter_sigmas: the sigma to generate the normal distribution of jitter amounts in each direction
        :param prob_slip: the probability of the slice shifting independently of all other slices
        :param prob_shift: the probability of the slice and all following slices shifting
        :param lcm_voxel_size: the least common voxel size of all the arrays in the request
        :return: the shift_array for the total_roi
        """
        # each row is one slice along jitter axis
        shift_array = np.zeros(shape=(jitter_axis_len, len(jitter_sigmas)), dtype=int)
        base_shift = np.zeros(shape=len(jitter_sigmas), dtype=int)
        assert(prob_slip + prob_shift <= 1)

        for jitter_axis_position in range(jitter_axis_len):
            r = random.random()
            slip = np.array([np.random.normal(scale=sigma / lcm_voxel_size[dimension])
                             for dimension, sigma in enumerate(jitter_sigmas)])
            slip = np.rint(slip).astype(int)
            slip = slip * np.array(lcm_voxel_size, dtype=int)

            if r <= prob_slip:
                shift_array[jitter_axis_position] = base_shift + slip
            elif r <= prob_slip + prob_shift:
                base_shift += slip
                shift_array[jitter_axis_position] = base_shift
            else:
                shift_array[jitter_axis_position] = base_shift

        return shift_array

    @staticmethod
    def compute_upstream_roi(request_roi, sub_shift_array):
        """ Compute the ROI to pass upstream for a specific item (array or points) in a request

        :param request_roi: the downstream ROI passed to the Jitter node
        :param sub_shift_array: the portion of the global shift array that should be used to shift the item
        :return: the expanded ROI to pass upstream
        """

        max_shift = Coordinate(sub_shift_array.max(axis=0))
        min_shift = Coordinate(sub_shift_array.min(axis=0))

        downstream_offset = request_roi.get_offset()
        upstream_offset = downstream_offset - max_shift
        upstream_shape = request_roi.get_shape() + max_shift - min_shift
        return Roi(offset=upstream_offset, shape=upstream_shape)
