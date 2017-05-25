import logging
import random

from .batch_filter import BatchFilter
from gunpowder.coordinate import Coordinate

logger = logging.getLogger(__name__)

class SimpleAugment(BatchFilter):

    def __init__(self, transpose_only_xy=True):
        self.transpose_only_xy = transpose_only_xy

    def prepare(self, request):

        self.total_roi = request.get_total_roi()
        self.dims = self.total_roi.dims()

        self.mirror = [ random.randint(0,1) for d in range(self.dims) ]
        if self.transpose_only_xy:
            assert self.dims==3, "Option transpose_only_xy only makes sense on 3D batches"
            t = [1,2]
            random.shuffle(t)
            self.transpose = (0,) + tuple(t)
        else:
            t = list(range(self.dims))
            random.shuffle(t)
            self.transpose = tuple(t)

        logger.debug("mirror = " + str(self.mirror))
        logger.debug("transpose = " + str(self.transpose))

        reverse_transpose = [0]*self.dims
        for d in range(self.dims):
            reverse_transpose[self.transpose[d]] = d

        logger.debug("downstream request = " + str(request))

        self.__transpose_request(request, reverse_transpose)
        self.__mirror_request(request, self.mirror)

        logger.debug("upstream request = " + str(request))

    def process(self, batch, request):

        mirror = tuple(
                slice(None, None, -1 if m else 1)
                for m in self.mirror
        )

        for (volume_type, volume) in batch.volumes.items():

            volume.data = volume.data[mirror]
            if self.transpose != (0,1,2):
                volume.data = volume.data.transpose(self.transpose)

            self.__mirror_roi(volume.roi, self.total_roi, self.mirror)
            self.__transpose_roi(volume.roi, self.transpose)

    def __mirror_request(self, request, mirror):

        for (volume_type, roi) in request.volumes.items():
            self.__mirror_roi(roi, self.total_roi, mirror)

    def __transpose_request(self, request, transpose):

        for (volume_type, roi) in request.volumes.items():
            self.__transpose_roi(roi, transpose)

    def __mirror_roi(self, roi, total_roi, mirror):

        total_roi_offset = total_roi.get_offset()
        total_roi_shape = total_roi.get_shape()

        roi_offset = roi.get_offset()
        roi_shape = roi.get_shape()

        roi_in_total_offset = roi_offset - total_roi_offset
        end_of_roi_in_total = roi_in_total_offset + roi_shape
        roi_in_total_offset_mirrored = total_roi_shape - end_of_roi_in_total
        roi_offset = Coordinate(
                total_roi_offset[d] + roi_in_total_offset_mirrored[d] if mirror[d] else roi_offset[d]
                for d in range(self.dims)
        )

        roi.set_offset(roi_offset)

    def __transpose_roi(self, roi, transpose):

        offset = roi.get_offset()
        shape = roi.get_shape()
        offset = tuple(offset[transpose[d]] for d in range(self.dims))
        shape = tuple(shape[transpose[d]] for d in range(self.dims))
        roi.set_offset(offset)
        roi.set_shape(shape)
