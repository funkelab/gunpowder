import logging
import random
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.coordinate import Coordinate

logger = logging.getLogger(__name__)

class SimpleAugment(BatchFilter):
    '''Randomly mirror and transpose all :class:`Arrays<Array>` and
    :class:`Graph` in a batch.

    Args:

        mirror_only (``list`` of ``int``, optional):

            If set, only mirror between the given axes. This is useful to
            exclude channels that have a set direction, like time.

        transpose_only (``list`` of ``int``, optional):

            If set, only transpose between the given axes. This is useful to
            limit the transpose to axes with the same resolution or to exclude
            non-spatial dimensions.
    '''

    def __init__(self, mirror_only=None, transpose_only=None):

        self.mirror_only = mirror_only
        self.transpose_only = transpose_only
        self.mirror_mask = None
        self.dims = None
        self.transpose_dims = None

    def setup(self):

        self.dims = self.spec.get_total_roi().dims()

        # mirror_mask and transpose_dims refer to the indices of the spatial
        # dimensions only, starting counting at 0 for the first spatial
        # dimension

        if self.mirror_only is None:
            self.mirror_mask = [ True ]*self.dims
        else:
            self.mirror_mask = [ d in self.mirror_only for d in range(self.dims) ]

        if self.transpose_only is None:
            self.transpose_dims = list(range(self.dims))
        else:
            self.transpose_dims = self.transpose_only

    def prepare(self, request):

        self.mirror = [
            random.randint(0,1)
            if self.mirror_mask[d] else 0
            for d in range(self.dims)
        ]

        t = list(self.transpose_dims)
        random.shuffle(t)
        self.transpose = list(range(self.dims))
        for o, n in zip(self.transpose_dims, t):
            self.transpose[o] = n

        logger.debug("mirror = " + str(self.mirror))
        logger.debug("transpose = " + str(self.transpose))

        reverse_transpose = [0]*self.dims
        for d in range(self.dims):
            reverse_transpose[self.transpose[d]] = d

        logger.debug("downstream request = " + str(request) +
                     "\nmirror = " + str(self.mirror) +
                     "\ntranspose = " + str(self.transpose))

        self.__transpose_request(request, reverse_transpose)
        self.__mirror_request(request, self.mirror)

        logger.debug("upstream request = " + str(request))

        return request

    def process(self, batch, request):

        # mirror and transpose ROIs of arrays & points in batch
        total_roi = batch.get_total_roi().copy()
        for collection_type in [batch.arrays, batch.points]:
            for (key, collector) in collection_type.items():
                if key not in request:
                    continue
                if collector.spec.roi is None:
                    continue
                logger.debug("total ROI: %s"%batch.get_total_roi())
                logger.debug("upstream %s ROI: %s"%(key, collector.spec.roi))
                self.__mirror_roi(collector.spec.roi, total_roi, self.mirror)
                logger.debug("mirrored %s ROI: %s"%(key,collector.spec.roi))
                self.__transpose_roi(collector.spec.roi, total_roi, self.transpose)
                logger.debug("transposed %s ROI: %s"%(key,collector.spec.roi))

        mirror = tuple(
                slice(None, None, -1 if m else 1)
                for m in self.mirror
        )
        # arrays
        for (array_key, array) in batch.arrays.items():

            if array_key not in request:
                continue

            if array.spec.nonspatial:
                continue

            num_channels = len(array.data.shape) - self.dims
            channel_slices = (slice(None, None),)*num_channels

            array.data = array.data[channel_slices + mirror]

            transpose = [t + num_channels for t in self.transpose]
            array.data = array.data = array.data.transpose(list(range(num_channels)) + transpose)

        # graphs
        total_roi_offset = batch.get_total_roi().get_offset()

        for (graph_key, graph) in batch.graphs.items():

            if graph_key not in request:
                continue

            for node in list(graph.nodes):
                # mirror
                location_in_total_offset = np.asarray(node.location) - total_roi_offset
                node.location[:] = np.asarray([batch.get_total_roi().get_end()[dim] - location_in_total_offset[dim]
                                                 if m else node.location[dim] for dim, m in enumerate(self.mirror)])
                # transpose
                if self.transpose != list(range(self.dims)):
                    for d in range(self.dims):
                        node.location[d] = \
                            location_in_total_offset[self.transpose[d]] + \
                            total_roi_offset[d]
                    logger.debug(node.location)

                # due to the mirroring, points at the lower boundary of the ROI
                # could fall on the upper one, which excludes them from the ROI
                if not graph.spec.roi.contains(node.location):
                    graph.remove_node(node)

    def __mirror_request(self, request, mirror):

        total_roi = request.get_total_roi().copy()
        for key, spec in request.items():
            if spec.roi is not None:
                self.__mirror_roi(spec.roi, total_roi, mirror)

    def __transpose_request(self, request, transpose):
        total_roi = request.get_total_roi().copy()
        for key, spec in request.items():
            if spec.roi is not None:
                self.__transpose_roi(spec.roi, total_roi, transpose)

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
        logger.debug("Mirror numbers for roi: " + str(roi)
                     + "\nMirror: " + str(mirror)
                     + "\tTranpose: " + str(self.transpose)
                     + "\ntotal roi: " + str(total_roi)
                     + "\nroi_in_total_offset: " + str(roi_in_total_offset)
                     + "\nend_of_roi_in_total: " + str(end_of_roi_in_total)
                     + "\nroi_in_total_offset_mirrored: " + str(roi_in_total_offset_mirrored)
                     + "\nroi_offset: " + str(roi_offset))

        roi.set_offset(roi_offset)

    def __transpose_roi(self, roi, total_roi, transpose):
        
        logger.debug(f"Original roi: {roi}")

        center = total_roi.get_center()
        logger.debug(f"Center: {center}")

        # Get distance from center, then transpose
        dist_to_center = center - roi.get_offset() 
        dist_to_center = Coordinate(dist_to_center[transpose[d]]
                                    for d in range(self.dims))
        logger.debug(f"dist_to_center: {dist_to_center}")

        # Using the tranposed distance to center, get the correct offset.
        new_offset = center - dist_to_center
        logger.debug(f"new_offset: {new_offset}")

        shape = tuple(roi.get_shape()[transpose[d]] for d in range(self.dims))
        roi.set_offset(new_offset)
        roi.set_shape(shape)
        logger.debug(f"Tranposed roi: {roi}")
