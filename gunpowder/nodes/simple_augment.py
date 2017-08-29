import logging
import random
import numpy as np

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
        # volumes
        for (volume_type, volume) in batch.volumes.items():

            volume.data = volume.data[mirror]
            if self.transpose != (0,1,2):
                volume.data = volume.data.transpose(self.transpose)
        # points
        total_roi_offset = self.total_roi.get_offset()
        for (points_type, points) in batch.points.items():

            for loc_id, syn_point in points.data.items():
                # mirror
                location_in_total_offset = np.asarray(syn_point.location) - total_roi_offset
                syn_point.location = np.asarray([self.total_roi.get_end()[dim] - location_in_total_offset[dim]
                                                 if m else syn_point.location[dim] for dim, m in enumerate(self.mirror)])
                # transpose
                if self.transpose != (0, 1, 2):
                    syn_point.location = np.asarray([syn_point.location[self.transpose[d]] for d in range(self.dims)])
        # volumes & points
        for collection_type in [batch.volumes, batch.points]:
            for (type, collector) in collection_type.items():
                logger.debug("total ROI: %s"%self.total_roi)
                logger.debug("upstream %s ROI: %s"%(type, collector.spec.roi))
                self.__mirror_roi(collector.spec.roi, self.total_roi, self.mirror)
                logger.debug("mirrored %s ROI: %s"%(type,collector.spec.roi))
                self.__transpose_roi(collector.spec.roi, self.transpose)
                logger.debug("transposed %s ROI: %s"%(type,collector.spec.roi))


    def __mirror_request(self, request, mirror):

        for identifier, spec in request.items():
            self.__mirror_roi(spec.roi, self.total_roi, mirror)

    def __transpose_request(self, request, transpose):

        for identifier, spec in request.items():
            self.__transpose_roi(spec.roi, transpose)

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
