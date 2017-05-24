import logging
import random

from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class SimpleAugment(BatchFilter):

    def __init__(self, transpose_only_xy=True):
        self.transpose_only_xy = transpose_only_xy

    def prepare(self, batch_spec):

        dims = batch_spec.input_roi.dims()

        self.mirror = [ random.randint(0,1) for d in range(dims) ]
        if self.transpose_only_xy:
            assert dims==3, "Option transpose_only_xy only makes sense on 3D batches"
            t = [1,2]
            random.shuffle(t)
            self.transpose = (0,) + tuple(t)
        else:
            t = list(range(dims))
            random.shuffle(t)
            self.transpose = tuple(t)

        logger.debug("downstream request input roi = " + str(batch_spec.input_roi))
        logger.debug("downstream request output roi = " + str(batch_spec.output_roi))
        logger.debug("mirror = " + str(self.mirror))
        logger.debug("transpose = " + str(self.transpose))

        reverse_transpose = [0]*dims
        for d in range(dims):
            reverse_transpose[self.transpose[d]] = d

        self.__transpose_spec(batch_spec, reverse_transpose)
        self.__mirror_spec(batch_spec, self.mirror)

        logger.debug("upstream request input roi = " + str(batch_spec.input_roi))
        logger.debug("upstream request output roi = " + str(batch_spec.output_roi))

    def process(self, batch):

        dims = batch.spec.input_roi.dims()

        mirror = tuple(
                slice(None, None, -1 if m else 1)
                for m in self.mirror
        )

        for (volume_type, volume) in batch.volumes.items():

            volume.data = volume.data[mirror]
            if self.transpose != (0,1,2):
                volume.data = volume.data.transpose(self.transpose)

        logger.debug("upstream batch shape = " + str(batch.spec.input_roi.get_shape()))
        self.__mirror_spec(batch.spec, self.mirror)
        self.__transpose_spec(batch.spec, self.transpose)
        logger.debug("downstream batch shape = " + str(batch.spec.input_roi.get_shape()))

    def __mirror_spec(self, spec, mirror):

        # this is a bit tricky: the offset and shape of input ROI stays the 
        # same, but the offset of the output ROI changes

        dims = spec.input_roi.dims()

        input_roi_offset = spec.input_roi.get_offset()
        input_roi_shape = spec.input_roi.get_shape()
        output_roi_offset = spec.output_roi.get_offset()
        output_roi_shape = spec.output_roi.get_shape()

        output_in_input_offset = tuple(
                output_roi_offset[d] - input_roi_offset[d]
                for d in range(dims)
        )
        end_of_output_in_input = tuple(
                output_in_input_offset[d] + output_roi_shape[d]
                for d in range(dims)
        )
        output_in_input_offset_mirrored = tuple(
                input_roi_shape[d] - end_of_output_in_input[d]
                for d in range(dims)
        )
        output_roi_offset = tuple(
                input_roi_offset[d] + output_in_input_offset_mirrored[d] if mirror[d] else output_roi_offset[d]
                for d in range(dims)
        )

        spec.output_roi.set_offset(output_roi_offset)

    def __transpose_spec(self, spec, transpose):

        self.__transpose_roi(spec.input_roi, transpose)
        self.__transpose_roi(spec.output_roi, transpose)

    def __transpose_roi(self, roi, transpose):

        dims = roi.dims()

        offset = roi.get_offset()
        shape = roi.get_shape()
        offset = tuple(offset[transpose[d]] for d in range(dims))
        shape = tuple(shape[transpose[d]] for d in range(dims))
        roi.set_offset(offset)
        roi.set_shape(shape)
