import multiprocessing
from freezable import Freezable
from roi import Roi

import logging
logger = logging.getLogger(__name__)

class BatchSpec(Freezable):
    '''A possibly partial specification of a batch.

    Used to request a batch from upstream batch providers. Will be refined on 
    the way up and set as the spec of the requested batch.
    '''

    __next_id = multiprocessing.Value('L')

    @staticmethod
    def get_next_id():
        with BatchSpec.__next_id.get_lock():
            next_id = BatchSpec.__next_id.value
            BatchSpec.__next_id.value += 1
        return next_id

    def __init__(self, input_shape, output_shape, input_offset=None, output_offset=None, resolution=None, with_gt=False, with_gt_mask=False, with_gt_affinities=False):

        if input_offset is None:
            input_offset = (0,)*len(input_shape)

        if output_offset is None:
            # assume output roi is centered in input roi
            output_offset = tuple(
                    int((input_shape[d] - output_shape[d])/2)
                    for d in range(len(input_shape))
            )

        self.input_roi = Roi(input_offset, input_shape)
        self.output_roi = Roi(output_offset, output_shape)
        self.resolution = resolution
        self.with_gt = with_gt
        self.with_gt_mask = with_gt_mask
        self.with_gt_affinities = with_gt_affinities
        self.id = BatchSpec.get_next_id()

        self.freeze()

        logger.debug("created new spec with id " + str(self.id))
