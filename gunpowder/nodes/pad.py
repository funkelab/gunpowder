import copy
import logging
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.roi import Roi
from gunpowder.coordinate import Coordinate

logger = logging.getLogger(__name__)

class Pad(BatchFilter):
    '''Add a constant intensity padding around arrays of another batch 
    provider. This is useful if your requested batches can be larger than what 
    your source provides.

    Args:

        pad_sizes(dict, ArrayKey -> [None,Coordinate]): Specifies the padding
            to be added to each array. If None, an infinite padding is added.
            If a Coordinate, this amount will be added to the ROI in the
            positive and negative direction.

        pad_values(dict, ArrayKey -> value or None): The values to report 
            inside the padding. If not given, 0 is used.
    '''

    def __init__(self, pad_sizes, pad_values=None):

        self.pad_sizes = pad_sizes
        if pad_values is None:
            self.pad_values = {}
        else:
            self.pad_values = pad_values

    def setup(self):

        for (array_key, pad_size) in self.pad_sizes.items():

            assert array_key in self.spec, "Asked to pad %s, but is not provided upstream."%array_key
            assert self.spec[array_key].roi is not None, "Asked to pad %s, but upstream provider doesn't have a ROI for it."%array_key

            spec = self.spec[array_key].copy()
            if pad_size is not None:
                spec.roi = spec.roi.grow(pad_size, pad_size)
            else:
                spec.roi.set_shape(None)
            self.updates(array_key, spec)

    def prepare(self, request):

        upstream_spec = self.get_upstream_provider().spec

        logger.debug("request: %s"%request)
        logger.debug("upstream spec: %s"%upstream_spec)

        for array_key in self.pad_sizes.keys():

            if array_key not in request:
                continue
            roi = request[array_key].roi

            # change request to fit into upstream spec
            request[array_key].roi = roi.intersect(upstream_spec[array_key].roi)

            if request[array_key].roi.empty():

                logger.warning("Requested %s ROI lies entirely outside of upstream ROI."%array_key)

                # ensure a valid request by asking for empty ROI
                request[array_key].roi = Roi(
                        upstream_spec[array_key].roi.get_offset(),
                        (0,)*upstream_spec[array_key].roi.dims()
                )

        logger.debug("new request: %s"%request)

    def process(self, batch, request):

        # restore requested batch size and ROI
        for (array_key, array) in batch.arrays.items():

            array.data = self.__expand(
                    array.data,
                    array.spec.roi/array.spec.voxel_size,
                    request[array_key].roi/array.spec.voxel_size,
                    self.pad_values[array_key] if array_key in self.pad_values else 0
            )
            array.spec.roi = request[array_key].roi

    def __expand(self, a, from_roi, to_roi, value):
        '''from_roi and to_roi should be in voxels.'''

        logger.debug("expanding array of shape %s from %s to %s"%(str(a.shape), from_roi, to_roi))

        b = np.zeros(to_roi.get_shape(), dtype=a.dtype)
        if value != 0:
            b[:] = value

        shift = tuple(-x for x in to_roi.get_offset())
        logger.debug("shifting 'from' by " + str(shift))
        a_in_b = from_roi.shift(shift).get_bounding_box()

        logger.debug("target shape is " + str(b.shape))
        logger.debug("target slice is " + str(a_in_b))

        b[a_in_b] = a

        return b
