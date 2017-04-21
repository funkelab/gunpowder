from batch_filter import BatchFilter
from random import randint

import logging
logger = logging.getLogger(__name__)

class RandomLocation(BatchFilter):
    '''Choses a batch at a random location in the bounding box of the upstream 
    provider.

    The random location is chosen such that the batch specs input roi lies 
    /inside/ the provder's roi.
    '''

    def setup(self):

        provider_spec = self.get_upstream_provider().get_spec()
        if provider_spec.roi.get_bounding_box() is None:
            raise RuntimeError("Can not draw random samples from a provider that does not have a bounding box.")

        self.roi = provider_spec.roi
        self.gt_roi = provider_spec.gt_roi

    def prepare(self, batch_spec):

        logger.info("original input ROI: %s"%batch_spec.input_roi)

        shape = batch_spec.input_roi.get_shape()

        for d in range(self.roi.dims()):
            assert self.roi.get_shape()[d] >= shape[d], "Requested shape %s does not fit into provided ROI %s."%(shape,self.roi)

        target_roi = self.roi

        # ensure that output ROI is inside GT ROI, if we know it
        if self.gt_roi is not None:

            # grow the GT ROI such that each input ROI contained in it has an 
            # output roi inside the original GT

            logger.info("ensuring output ROI is within GT roi")

            output_offset = batch_spec.output_roi.get_offset()
            output_shape = batch_spec.output_roi.get_shape()
            input_offset = batch_spec.input_roi.get_offset()
            input_shape = batch_spec.input_roi.get_shape()
            gt_shape = self.gt_roi.get_shape()

            # set a new offset for the expanded GT ROI
            diff = tuple(input_offset[d] - output_offset[d] for d in range(self.gt_roi.dims()))
            gt_roi_expanded = self.gt_roi.shift(diff)

            # increase the size
            gt_roi_shape = tuple(
                    gt_shape[d] + (input_shape[d] - output_shape[d])
                    for d in range(self.gt_roi.dims())
            )
            gt_roi_expanded.set_shape(gt_roi_shape)

            logger.info("GT ROI: " + str(self.gt_roi))
            logger.info("target ROI guaranteeing output ROI to be in GT ROI: " + str(gt_roi_expanded))
            logger.info("current target ROI for input ROI: " + str(target_roi))

            target_roi = gt_roi_expanded.intersect(target_roi)

            logger.info("intersection of valid ROIs: " + str(target_roi))

        target_bb = target_roi.get_bounding_box()

        current_offset = batch_spec.input_roi.get_offset()
        new_offset = tuple(
                randint(target_bb[d].start, target_bb[d].stop - shape[d])
                for d in range(len(shape))
        )
        diff = tuple(
                new_offset[d] - current_offset[d]
                for d in range(batch_spec.input_roi.dims())
        )

        batch_spec.input_roi = batch_spec.input_roi.shift(diff)
        batch_spec.output_roi = batch_spec.output_roi.shift(diff)

        logger.info("target ROI: %s"%target_roi)
        logger.info("current offset: %s"%str(current_offset))
        logger.info("new random offset: %s"%str(new_offset))
        logger.info("new input ROI: %s"%batch_spec.input_roi)
        logger.info("new output ROI: %s"%batch_spec.output_roi)

        assert self.roi.contains(batch_spec.input_roi)
        if self.gt_roi is not None:
            assert self.gt_roi.contains(batch_spec.output_roi)

    def process(self, batch):
        pass
