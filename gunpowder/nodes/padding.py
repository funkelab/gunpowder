from batch_filter import BatchFilter
from gunpowder.roi import Roi
from gunpowder.coordinate import Coordinate
from gunpowder.volume import VolumeType
import copy
import numpy as np

import logging
logger = logging.getLogger(__name__)

class Padding(BatchFilter):
    '''Add a constant intensity padding around a batch provider. This is useful 
    if your requested batches can be larger than what your source provides.
    '''

    def __init__(self, padding=None, outside_raw_value=0):
        '''If padding is None (default), implements an infinite padding. In this 
        case, the reported provider spec is the same as the upstream provider 
        spec, but access outside the ROI is permitted.

        If padding is a Coordinate, this amount will be added to the upstream 
        ROI in the positive and negative direction, and the thus grown ROI will 
        be reported downstream as the new provider spec. Futhermore, access 
        outside of the grown ROI will result in an exception.
        '''
        if padding is not None:
            self.padding = Coordinate(padding)
        else:
            self.padding = None
        self.outside_raw_value = outside_raw_value

    def setup(self):
        self.upstream_spec = self.get_upstream_provider().get_spec()
        self.spec = copy.deepcopy(self.upstream_spec)

        assert self.spec.roi.get_bounding_box() is not None, "Padding can only be applied after a source that provides a bounding box."

        if self.padding is not None:
            self.spec.roi = self.upstream_spec.roi.grow(self.padding, self.padding)

        logger.debug("Upstream roi: " + str(self.upstream_spec.roi))
        logger.debug("Provided roi:" + str(self.spec.roi))

    def get_spec(self):
        return self.spec

    def prepare(self, batch_spec):

        if self.padding is not None:
            if not self.spec.roi.intersects(batch_spec.input_roi):
                raise RuntimeError("Input ROI of batch " + str(batch_spec.input_roi) + " lies outside of my ROI " + str(self.spec.roi))

        # remember request batch spec
        self.request_batch_spec = copy.deepcopy(batch_spec)

        # change batch spec to fit into upstream spec
        logger.debug("request: %s"%batch_spec)
        logger.debug("upstream ROI: %s"%self.upstream_spec.roi)
        batch_spec.input_roi = batch_spec.input_roi.intersect(self.upstream_spec.roi)
        batch_spec.output_roi = batch_spec.output_roi.intersect(self.upstream_spec.roi)

        if batch_spec.input_roi is None or batch_spec.output_roi is None:
            logger.warning("Requested batch lies entirely outside of upstream ROI.")
            batch_spec.input_roi = Roi(self.upstream_spec.roi.get_offset(), (0,)*self.upstream_spec.roi.dims())
            batch_spec.output_roi = Roi(self.upstream_spec.roi.get_offset(), (0,)*self.upstream_spec.roi.dims())

        logger.debug("new request: %s"%batch_spec)

    def process(self, batch):

        # restore requested batch size
        for (volume_type, volume) in batch.volumes:
            if volume_type == VolumeType.RAW:
                volume.data = self.__expand(volume.data, batch.spec.input_roi, self.request_batch_spec.input_roi, self.outside_raw_value)
            else:
                volume.data = self.__expand(batch.gt, batch.spec.output_roi, self.request_batch_spec.output_roi, 0)

        batch.spec = self.request_batch_spec

    def __expand(self, a, from_roi, to_roi, value):

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
