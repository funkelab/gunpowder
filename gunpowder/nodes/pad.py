import copy
import logging
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.array import ArrayKey
from gunpowder.coordinate import Coordinate
from gunpowder.points import PointsKey
from gunpowder.roi import Roi

logger = logging.getLogger(__name__)

class Pad(BatchFilter):
    '''Add a constant intensity padding around arrays of another batch 
    provider. This is useful if your requested batches can be larger than what 
    your source provides.

    Args:

        key (:class:`ArrayKey` or :class:`PointsKey`):

            The array or points set to pad.

        size (:class:`Coordinate` or ``None``):

            The padding to be added. If None, an infinite padding is added. If
            a coordinate, this amount will be added to the ROI in the positive
            and negative direction.

        value (scalar or ``None``):

            The value to report inside the padding. If not given, 0 is used.
            Only used for :class:`Array<Arrays>`.
    '''

    def __init__(self, key, size, value=None):

        self.key = key
        self.size = size
        self.value = value

    def setup(self):

        assert self.key in self.spec, (
            "Asked to pad %s, but is not provided upstream."%self.key)
        assert self.spec[self.key].roi is not None, (
            "Asked to pad %s, but upstream provider doesn't have a ROI for "
            "it."%self.key)

        spec = self.spec[self.key].copy()
        if self.size is not None:
            spec.roi = spec.roi.grow(self.size, self.size)
        else:
            spec.roi.set_shape(None)
        self.updates(self.key, spec)

    def prepare(self, request):

        upstream_spec = self.get_upstream_provider().spec

        logger.debug("request: %s"%request)
        logger.debug("upstream spec: %s"%upstream_spec)

        if self.key not in request:
            return

        roi = request[self.key].roi.copy()

        # change request to fit into upstream spec
        request[self.key].roi = roi.intersect(upstream_spec[self.key].roi)

        if request[self.key].roi.empty():

            logger.warning(
                "Requested %s ROI %s lies entirely outside of upstream "
                "ROI %s.", self.key, roi, upstream_spec[self.key].roi)

            # ensure a valid request by asking for empty ROI
            request[self.key].roi = Roi(
                    upstream_spec[self.key].roi.get_offset(),
                    (0,)*upstream_spec[self.key].roi.dims()
            )

        logger.debug("new request: %s"%request)

    def process(self, batch, request):

        if self.key not in request:
            return

        # restore requested batch size and ROI
        if isinstance(self.key, ArrayKey):

            array = batch.arrays[self.key]
            array.data = self.__expand(
                    array.data,
                    array.spec.roi/array.spec.voxel_size,
                    request[self.key].roi/array.spec.voxel_size,
                    self.value if self.value else 0
            )
            array.spec.roi = request[self.key].roi

        else:

            points = batch.points[self.key]
            points.spec.roi = request[self.key].roi

    def __expand(self, a, from_roi, to_roi, value):
        '''from_roi and to_roi should be in voxels.'''

        logger.debug(
            "expanding array of shape %s from %s to %s",
            str(a.shape), from_roi, to_roi)

        num_channels = len(a.shape) - from_roi.dims()
        channel_shapes = a.shape[:num_channels]

        b = np.zeros(channel_shapes + to_roi.get_shape(), dtype=a.dtype)
        if value != 0:
            b[:] = value

        shift = tuple(-x for x in to_roi.get_offset())
        logger.debug("shifting 'from' by " + str(shift))
        a_in_b = from_roi.shift(shift).to_slices()

        logger.debug("target shape is " + str(b.shape))
        logger.debug("target slice is " + str(a_in_b))

        b[(slice(None),)*num_channels + a_in_b] = a

        return b
