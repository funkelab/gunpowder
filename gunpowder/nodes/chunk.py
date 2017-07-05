import copy
import logging
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.volume import VolumeTypes, Volume

logger = logging.getLogger(__name__)

class Chunk(BatchFilter):
    '''Assemble a large batch by requesting smaller chunks upstream.
    '''

    def __init__(self, chunk_spec):

        self.chunk_spec_template = chunk_spec
        self.dims = self.chunk_spec_template.volumes[self.chunk_spec_template.volumes.keys()[0]].dims()

        for volume_type in self.chunk_spec_template.volumes:
            assert self.dims == self.chunk_spec_template.volumes[volume_type].dims(),\
                "Volumes of different dimensionalities cannot be handled by chunk"


    def provide(self, request):

        logger.info("batch with spec " + str(request) + " requested")

        # minimal stride is smallest shape in template volumes because they are all centered
        min_stride = self.chunk_spec_template.get_common_roi().get_shape()

        # initial shift required per volume to be at beginning of its requested roi
        all_initial_offsets = []
        for (volume_type, roi) in self.chunk_spec_template.volumes.items():
            all_initial_offsets.append(request.volumes[volume_type].get_begin() - roi.get_begin())
        begin = np.min(all_initial_offsets, axis=0)

        # max offsets required per volume to cover their entire requested roi
        all_max_offsets = []
        for (volume_type, roi) in self.chunk_spec_template.volumes.items():
            all_max_offsets.append(request.volumes[volume_type].get_end()-self.chunk_spec_template.volumes[volume_type].get_shape())
        end = np.max(all_max_offsets, axis=0) + min_stride

        batch = None
        offset = np.array(begin)
        while (offset < end).all():

            # create a copy of the requested batch spec
            chunk_request = copy.deepcopy(request)
            max_strides = []
            # change size and offset of the batch spec
            for volume_type, roi in self.chunk_spec_template.volumes.items():
                chunk_request.volumes[volume_type] = roi + Coordinate(offset)
                # adjust stride to be as large as possible. Chunk roi lies either:
                #   in front and within roi, then max stride shifts chunk roi to begin of request roi
                #   behind requested roi, ten max stride shifts chunk roi to end of ALL rois in request
                # finally, clip max_stride s.t. it is not smaller than min_stride
                max_stride = np.zeros([3])
                for dim in range(roi.dims()):
                    if request.volumes[volume_type].get_end()[dim] > chunk_request.volumes[volume_type].get_end()[dim]:
                        max_stride[dim] = request.volumes[volume_type].get_begin()[dim] - chunk_request.volumes[volume_type].get_begin()[dim]
                    else:
                        max_stride[dim] = end[dim] - offset[dim]
                max_strides.append(max_stride.clip(min_stride))

            stride = np.min(max_strides, axis=0)

            logger.info("requesting chunk " + str(chunk_request))

            # get a chunk
            chunk = self.get_upstream_provider().request_batch(chunk_request)

            if batch is None:
                batch = self.__setup_batch(request, chunk)

            # fill returned chunk into batch
            for (volume_type, volume) in chunk.volumes.items():
                self.__fill(batch.volumes[volume_type].data, volume.data,
                            request.volumes[volume_type], volume.roi)

            for d in range(self.dims):
                offset[d] += stride[d]
                if offset[d] >= end[d]:
                    if d == self.dims - 1:
                        break
                    offset[d] = begin[d]
                else:
                    break

        return batch


    def __setup_batch(self, request, chunk_batch):

        batch = Batch()
        for (volume_type, roi) in request.volumes.items():
            if volume_type == VolumeTypes.PRED_AFFINITIES or volume_type == VolumeTypes.GT_AFFINITIES:
                shape = (3,)+ roi.get_shape()
            else:
                shape = roi.get_shape()

            batch.volumes[volume_type] = Volume(data=np.zeros(shape),
                                                roi=roi,
                                                resolution=chunk_batch.volumes[VolumeTypes.RAW].resolution)
        return batch


    def __fill(self, a, b, roi_a, roi_b):
        logger.debug("filling " + str(roi_b) + " into " + str(roi_a))

        common_roi = roi_a.intersect(roi_b)
        if common_roi is None:
            return

        common_in_a_roi = common_roi - roi_a.get_offset()
        common_in_b_roi = common_roi - roi_b.get_offset()

        slices_a = common_in_a_roi.get_bounding_box()
        slices_b = common_in_b_roi.get_bounding_box()

        if len(a.shape) > len(slices_a):
            slices_a = (slice(None),)*(len(a.shape) - len(slices_a)) + slices_a
            slices_b = (slice(None),)*(len(b.shape) - len(slices_b)) + slices_b

        a[slices_a] = b[slices_b]
