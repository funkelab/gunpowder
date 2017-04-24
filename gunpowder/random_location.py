from batch_filter import BatchFilter
from random import randint
from coordinate import Coordinate

import logging
logger = logging.getLogger(__name__)

class RandomLocation(BatchFilter):
    '''Choses a batch at a random location in the bounding box of the upstream 
    provider.

    The random location is chosen such that the batch specs input roi lies 
    entirely inside the provder's roi.

    If the provider has a ground-truth ROI, the location is chosen such that the 
    center of the batch spec's output ROI is inside the provider's ground-truth 
    ROI.
    '''

    def setup(self):

        provider_spec = self.get_upstream_provider().get_spec()
        if provider_spec.roi.get_bounding_box() is None:
            raise RuntimeError("Can not draw random samples from a provider that does not have a bounding box.")

        self.roi = provider_spec.roi
        self.gt_roi = provider_spec.gt_roi

    def prepare(self, batch_spec):

        logger.info("requested input ROI: %s"%batch_spec.input_roi)
        logger.info("requested output ROI: %s"%batch_spec.output_roi)

        shape = batch_spec.input_roi.get_shape()
        for d in range(self.roi.dims()):
            assert self.roi.get_shape()[d] >= shape[d], "Requested shape %s does not fit into provided ROI %s."%(shape,self.roi)

        target_roi = self.roi
        logger.info("valid target ROI to fit input request: " + str(target_roi))

        # ensure that output center is inside GT ROI, if we know it
        if self.gt_roi is not None:

            logger.info("GT ROI is set, I will ensure that center of output is inside GT ROI")

            # get output center
            output_center = batch_spec.output_roi.get_center()

            # from input_roi min to center
            # = amount to grow GT ROI in negative direction
            grow_neg = output_center - batch_spec.input_roi.get_begin()

            # from center to input_roi max
            # = amount to grow GT ROI in positive direction
            grow_pos = batch_spec.input_roi.get_end() - output_center

            logger.info("center of output request is at " + str(output_center))
            logger.info("growing GT roi by " + str(grow_neg) + " and " + str(grow_pos))

            # grow the GT ROI
            expanded_gt_roi = self.gt_roi.grow(grow_neg, grow_pos)

            logger.info("original GT ROI: " + str(self.gt_roi))
            logger.info("expanded GT ROI: " + str(expanded_gt_roi))

            target_roi = expanded_gt_roi.intersect(target_roi)

            logger.info("intersection of valid ROIs: " + str(target_roi))

        # shrink target ROI, such that it contains only valid offset positions 
        # for input ROI
        target_roi = target_roi.grow(None, -batch_spec.input_roi.get_shape())

        logger.info("valid starting points for input request in " + str(target_roi))

        # select a random point inside ROI
        random_offset = Coordinate(
                randint(begin, end-1)
                for begin, end in zip(target_roi.get_begin(), target_roi.get_end())
        )

        logger.info("random starting point: " + str(random_offset))

        # shift input and output ROI
        diff = random_offset - batch_spec.input_roi.get_offset()
        batch_spec.input_roi = batch_spec.input_roi.shift(diff)
        batch_spec.output_roi = batch_spec.output_roi.shift(diff)

        logger.info("new input ROI: %s"%batch_spec.input_roi)
        logger.info("new output ROI: %s"%batch_spec.output_roi)
        logger.info("center of output ROI: " + str(batch_spec.output_roi.get_center()))

        assert self.roi.contains(batch_spec.input_roi)
        if self.gt_roi is not None:
            assert self.gt_roi.contains(batch_spec.output_roi.get_center())

    def process(self, batch):
        pass
