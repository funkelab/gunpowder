from batch_filter import BatchFilter
from roi import Roi
import copy
import numpy as np

import logging
logger = logging.getLogger(__name__)

class Padding(BatchFilter):
    '''Add a constant intensity padding around a batch provider. This is useful 
    if your requested batches can be larger than what your source provides.
    '''

    def __init__(self, padding, outside_raw_value=0):
        self.padding = padding
        self.outside_raw_value = outside_raw_value

    def setup(self):
        self.upstream_spec = self.get_upstream_provider().get_spec()
        self.spec = copy.deepcopy(self.upstream_spec)

        assert self.spec.roi.get_bounding_box() is not None, "Padding can only be applied after a source that provides a bounding box."

        offset = tuple(
                self.upstream_spec.roi.get_offset()[d] - self.padding[d]
                for  d in range(self.upstream_spec.roi.dims())
        )
        shape = tuple(
                self.upstream_spec.roi.get_shape()[d] + 2*self.padding[d]
                for  d in range(self.upstream_spec.roi.dims())
        )
        self.spec.roi = Roi(offset, shape)

        logger.info("Upstream roi: " + str(self.upstream_spec.roi))
        logger.info("Provided roi:" + str(self.spec.roi))

    def get_spec(self):
        return self.spec

    def prepare(self, batch_spec):

        # remember request batch spec
        self.request_batch_spec = copy.deepcopy(batch_spec)

        # change batch spec to fit into upstream spec
        logger.debug("previous input request ROI: %s"%batch_spec.input_roi)
        logger.debug("upstream ROI: %s"%self.upstream_spec.roi)
        batch_spec.input_roi = batch_spec.input_roi.intersect(self.upstream_spec.roi)
        batch_spec.output_roi = batch_spec.output_roi.intersect(self.upstream_spec.roi)

        if batch_spec.input_roi is None or batch_spec.output_roi is None:
            logger.warning("Requested batch lies entirely in padded region.")
            batch_spec.input_roi = Roi(self.upstream_spec.roi.get_offset(), (0,)*self.upstream_spec.roi.dims())
            batch_spec.output_roi = Roi(self.upstream_spec.roi.get_offset(), (0,)*self.upstream_spec.roi.dims())

        logger.debug("new input request ROI: %s"%batch_spec.input_roi)

    def process(self, batch):

        # restore requested batch size
        batch.raw = self.__expand(batch.raw, batch.spec.input_roi, self.request_batch_spec.input_roi, self.outside_raw_value)
        if batch.gt is not None:
            batch.gt = self.__expand(batch.gt, batch.spec.output_roi, self.request_batch_spec.output_roi, 0)
        if batch.gt_mask is not None:
            batch.gt_mask = self.__expand(batch.gt_mask, batch.spec.output_roi, self.request_batch_spec.output_roi, 0)

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
