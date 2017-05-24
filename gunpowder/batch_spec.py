import logging
import multiprocessing

from .coordinate import Coordinate
from .freezable import Freezable
from .roi import Roi

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

    def __init__(
            self,
            input_shape,
            output_shape,
            input_offset=None,
            output_offset=None,
            resolution=None,
            with_volumes=[]):

        input_shape = Coordinate(input_shape)
        output_shape = Coordinate(output_shape)

        if input_offset is None:
            input_offset = Coordinate((0,)*input_shape.dims())
        else:
            input_offset = Coordinate(input_offset)

        if output_offset is None:
            # assume output roi is centered in input roi
            output_offset = input_offset + (input_shape - output_shape)/2
        else:
            output_offset = Coordinate(output_offset)

        if resolution is not None:
            resolution = Coordinate(resolution)

        self.input_roi = Roi(input_offset, input_shape)
        self.output_roi = Roi(output_offset, output_shape)
        self.resolution = resolution
        self.with_volumes = with_volumes
        self.id = BatchSpec.get_next_id()

        self.freeze()

        logger.debug("created new spec with id " + str(self.id))

    def __repr__(self):

        r  = "input ROI   : " + str(self.input_roi) + "\n"
        r += "output ROI  : " + str(self.output_roi) + "\n"
        r += "resolution  : " + str(self.resolution) + "\n"
        r += "with volumes: " + str(self.with_volumes) + "\n"
        r += "ID          : " + str(self.id)

        return r
