import logging
import random
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.coordinate import Coordinate

logger = logging.getLogger(__name__)

class SimpleAugment(BatchFilter):
    '''Randomly mirror and transpose all :class:`Arrays<Array>` and
    :class:`Points` in a batch.

    Args:

        mirror_only (``list`` of ``int``, optional):

            If set, only mirror between the given axes. This is useful to
            exclude channels that have a set direction, like time.

        transpose_only (``list`` of ``int``, optional):

            If set, only transpose between the given axes. This is useful to
            limit the transpose to axes with the same resolution or to exclude
            non-spatial dimensions.
    '''

    def __init__(self, mirror_only=None, transpose_only=None):

        self.mirror_only = mirror_only
        self.transpose_only = transpose_only
        self.mirror_mask = None
        self.dims = None
        self.transpose_dims = None

    def setup(self):

        self.dims = self.spec.get_total_roi().dims()

        # mirror_mask and transpose_dims refer to the indices of the spatial
        # dimensions only, starting counting at 0 for the first spatial
        # dimension

        if self.mirror_only is None:
            self.mirror_mask = [ True ]*self.dims
        else:
            self.mirror_mask = [ d in self.mirror_only for d in range(self.dims) ]

        if self.transpose_only is None:
            self.transpose_dims = list(range(self.dims))
        else:
            self.transpose_dims = self.transpose_only

    def prepare(self, request):
        random.seed(request.random_seed)

        self.total_roi = request.get_total_roi()

        self.mirror = [
            random.randint(0,1)
            if self.mirror_mask[d] else 0
            for d in range(self.dims)
        ]

        t = list(self.transpose_dims)
        random.shuffle(t)
        self.transpose = list(range(self.dims))
        for o, n in zip(self.transpose_dims, t):
            self.transpose[o] = n

        logger.debug("mirror = " + str(self.mirror))
        logger.debug("transpose = " + str(self.transpose))

        reverse_transpose = [0]*self.dims
        for d in range(self.dims):
            reverse_transpose[self.transpose[d]] = d

        logger.debug("downstream request = " + str(request))

        self.__transpose_request(request, reverse_transpose)
        self.__mirror_request(request, self.mirror)

        logger.debug("upstream request = " + str(request))

    def process(self, batch, request):

        # mirror and transpose ROIs of arrays & points in batch
        for collection_type in [batch.arrays, batch.points]:
            for (key, collector) in collection_type.items():
                if key not in request:
                    continue
                if collector.spec.roi is None:
                    continue
                logger.debug("total ROI: %s"%self.total_roi)
                logger.debug("upstream %s ROI: %s"%(key, collector.spec.roi))
                self.__mirror_roi(collector.spec.roi, self.total_roi, self.mirror)
                logger.debug("mirrored %s ROI: %s"%(key,collector.spec.roi))
                self.__transpose_roi(collector.spec.roi, self.transpose)
                logger.debug("transposed %s ROI: %s"%(key,collector.spec.roi))

        mirror = tuple(
                slice(None, None, -1 if m else 1)
                for m in self.mirror
        )
        # arrays
        for (array_key, array) in batch.arrays.items():

            if array_key not in request:
                continue

            if array.spec.nonspatial:
                continue

            num_channels = len(array.data.shape) - self.dims
            channel_slices = (slice(None, None),)*num_channels

            array.data = array.data[channel_slices + mirror]

            transpose = [t + num_channels for t in self.transpose]
            array.data = array.data.transpose([0]*num_channels + transpose)

        # points
        total_roi_offset = self.total_roi.get_offset()
        for (points_key, points) in batch.points.items():

            if points_key not in request:
                continue

            for loc_id, syn_point in list(points.data.items()):
                # mirror
                location_in_total_offset = np.asarray(syn_point.location) - total_roi_offset
                syn_point.location = np.asarray([self.total_roi.get_end()[dim] - location_in_total_offset[dim]
                                                 if m else syn_point.location[dim] for dim, m in enumerate(self.mirror)])
                # transpose
                if self.transpose != tuple(range(self.dims)):
                    syn_point.location = np.asarray([syn_point.location[self.transpose[d]] for d in range(self.dims)])

                # due to the mirroring, points at the lower boundary of the ROI
                # could fall on the upper one, which excludes them from the ROI
                if not points.spec.roi.contains(syn_point.location):
                    del points.data[loc_id]


    def __mirror_request(self, request, mirror):

        for key, spec in request.items():
            if spec.roi is not None:
                self.__mirror_roi(spec.roi, self.total_roi, mirror)

    def __transpose_request(self, request, transpose):

        for key, spec in request.items():
            if spec.roi is not None:
                self.__transpose_roi(spec.roi, transpose)

    def __mirror_roi(self, roi, total_roi, mirror):

        total_roi_offset = total_roi.get_offset()
        total_roi_shape = total_roi.get_shape()

        roi_offset = roi.get_offset()
        roi_shape = roi.get_shape()

        roi_in_total_offset = roi_offset - total_roi_offset
        end_of_roi_in_total = roi_in_total_offset + roi_shape
        roi_in_total_offset_mirrored = total_roi_shape - end_of_roi_in_total
        roi_offset = Coordinate(
                total_roi_offset[d] + roi_in_total_offset_mirrored[d] if mirror[d] else roi_offset[d]
                for d in range(self.dims)
        )

        roi.set_offset(roi_offset)

    def __transpose_roi(self, roi, transpose):

        offset = roi.get_offset()
        shape = roi.get_shape()
        offset = tuple(offset[transpose[d]] for d in range(self.dims))
        shape = tuple(shape[transpose[d]] for d in range(self.dims))
        roi.set_offset(offset)
        roi.set_shape(shape)
