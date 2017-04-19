import random
from batch_filter import BatchFilter

import logging
logger = logging.getLogger(__name__)

class SimpleAugment(BatchFilter):

    def __init__(self, transpose_only_xy=True):
        self.transpose_only_xy = transpose_only_xy

    def prepare(self, batch_spec):

        self.mirror = [ random.randint(0,1) for d in range(len(batch_spec.shape)) ]
        if self.transpose_only_xy:
            assert len(batch_spec.shape)==3, "Option transpose_only_xy only makes sense on 3D batches"
            t = [1,2]
            random.shuffle(t)
            self.transpose = (0,) + tuple(t)
        else:
            t = list(range(len(batch_spec.shape)))
            random.shuffle(t)
            self.transpose = tuple(t)

        logger.debug("SimpleAugment: downstream request shape = " + str(batch_spec.shape))
        logger.debug("SimpleAugment: mirror = " + str(self.mirror))
        logger.debug("SimpleAugment: transpose = " + str(self.transpose))

        batch_spec.shape = tuple(batch_spec.shape[self.transpose[d]] for d in range(len(batch_spec.shape)))

        logger.debug("SimpleAugment: upstream request shape = " + str(batch_spec.shape))

    def process(self, batch):

        mirror = tuple(
                slice(None, None, -1 if m else 1)
                for m in self.mirror
        )

        batch.raw = batch.raw[mirror]
        if batch.gt is not None:
            batch.gt = batch.gt[mirror]
        if batch.gt_mask is not None:
            batch.gt_mask = batch.gt_mask[mirror]

        if self.transpose != (0,1,2):
            batch.raw = batch.raw.transpose(self.transpose)
            if batch.gt is not None:
                batch.gt = batch.gt.transpose(self.transpose)
            if batch.gt_mask is not None:
                batch.gt_mask = batch.gt_mask.transpose(self.transpose)
