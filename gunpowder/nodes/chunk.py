import copy
import logging
import multiprocessing
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.producer_pool import ProducerPool
from gunpowder.points import PointsTypes, Points
from gunpowder.roi import Roi
from gunpowder.volume import VolumeTypes, Volume

logger = logging.getLogger(__name__)

class Chunk(BatchFilter):
    '''Assemble a large batch by requesting smaller chunks upstream.
    '''

    def __init__(self, chunk_spec, cache_size=50, num_workers=1):

        self.chunk_spec_template = chunk_spec
        self.cache_size          = cache_size
        self.num_workers         = num_workers
        self.dims = self.chunk_spec_template.volume_specs[self.chunk_spec_template.volume_specs.keys()[0]].roi.dims()

        for identifier, spec in self.chunk_spec_template.items():
            assert self.dims == spec.roi.dims(),\
                "Rois of different dimensionalities cannot be handled by chunk"

    def setup(self):

        for identifier, spec in self.spec.volume_specs.items():
            assert spec.roi is not None, (
                "The ROI of volume %s is not specified, Chunk does not know "
                "what to do."%identifier)

    def teardown(self):
        if self.num_workers > 1:
            self.workers.stop()

    def provide(self, request):

        self.request = copy.deepcopy(request)

        # prepare and queue all requests required by chunk
        self.__prepare_requests()

        # setup several workers if num_workers > 1
        if self.num_workers > 1:
            self.workers = ProducerPool([ lambda i=i: self.__run_worker(i) for i in range(self.num_workers) ], queue_size=self.cache_size)
            self.workers.start()

        batch = None
        for i in range(self.num_requests):
            if self.num_workers > 1:
                chunk = self.workers.get()
            else:
                chunk_request = self.requests.get()
                chunk = self.get_upstream_provider().request_batch(chunk_request)

            if batch is None:
                batch = self.__setup_batch(request, chunk)

            # fill returned chunk into batch
            for (volume_type, volume) in chunk.volumes.items():
                self.__fill(batch.volumes[volume_type].data, volume.data,
                            request.volume_specs[volume_type].roi, volume.spec.roi, self.spec[volume_type].voxel_size)

            for (points_type, points) in chunk.points.items():
                self.__fill_points(batch.points[points_type].data, points.data,
                                   request.points_specs[points_type].roi, points.roi)

        return batch

    def __setup_batch(self, request, chunk_batch):

        batch = Batch()
        for (volume_type, spec) in request.volume_specs.items():
            roi = spec.roi
            voxel_size = self.spec[volume_type].voxel_size

            # get the 'non-spatial' shape of the chunk-batch
            # and append the shape of the request to it
            volume = chunk_batch.volumes[volume_type]
            shape = volume.data.shape[:-roi.dims()]
            shape += (roi.get_shape() // voxel_size)

            spec = self.spec[volume_type].copy()
            spec.roi = roi
            batch.volumes[volume_type] = Volume(data=np.zeros(shape),
                                                spec=spec)

        for (points_type, spec) in request.points_specs.items():
            roi = spec.roi
            spec = self.spec[points_type].copy()
            spec.roi = roi
            batch.points[points_type] = Points(data={}, spec=spec)

        return batch

    def __fill(self, a, b, roi_a, roi_b, voxel_size):
        logger.debug("filling " + str(roi_b) + " into " + str(roi_a))

        roi_a = roi_a // voxel_size
        roi_b = roi_b // voxel_size

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

    def __fill_points(self, a, b, roi_a, roi_b):
        logger.debug("filling points of " + str(roi_b) + " into points of" + str(roi_a))

        common_roi = roi_a.intersect(roi_b)
        if common_roi is None:
            return

        # find max point_id in a so far
        max_point_id = 0
        for point_id, point in a.items():
            if point_id > max_point_id:
                max_point_id = point_id

        for point_id, point in b.items():
            if roi_a.contains(Coordinate(point.location)):
                a[point_id + max_point_id] = point

    def __run_worker(self, i):
        chosen_request = self.requests.get()
        return self.get_upstream_provider().request_batch(chosen_request)

    def __prepare_requests(self):
        ''' prepare all request which chunk will require and store them in queue '''
        self.requests = multiprocessing.Queue(maxsize=0)

        logger.info("batch with spec " + str(self.request) + " requested")

        # initial offset required per volume to be at beginning of its requested roi
        all_initial_offsets = []
        for chunk_spec_type, request_type in zip([self.chunk_spec_template.volume_specs, self.chunk_spec_template.points_specs],
                                                 [self.request.volume_specs, self.request.points_specs]):
            for (type, spec) in chunk_spec_type.items():
                roi = spec.roi
                all_initial_offsets.append(request_type[type].roi.get_begin() - roi.get_offset())
        initial_offset = np.min(all_initial_offsets, axis=0)

        # max offsets required per volume to cover their entire requested roi
        all_max_offsets = []
        for chunk_spec_type, request_type in zip([self.chunk_spec_template.volume_specs, self.chunk_spec_template.points_specs],
                                                 [self.request.volume_specs, self.request.points_specs]):
            for (type, spec) in chunk_spec_type.items():
                roi = spec.roi
                all_max_offsets.append(request_type[type].roi.get_end() - chunk_spec_type[type].roi.get_shape()-chunk_spec_type[type].roi.get_offset())
        final_offset = np.max(all_max_offsets, axis=0)

        # check for all VolumeTypes and PointsTypes in template if requests of chunk can be provided
        for chunk_spec_type, provider_spec_type in zip([self.chunk_spec_template.volume_specs, self.chunk_spec_template.points_specs],
                                                        [self.spec.volume_specs, self.spec.points_specs]):
            for type, spec in chunk_spec_type.items():
                roi = spec.roi
                complete_chunk_roi = Roi(initial_offset+roi.get_offset(),
                                         (final_offset - initial_offset) + roi.get_shape())
                assert provider_spec_type[type].roi.contains(complete_chunk_roi), "Request with current chunk template" \
                        " request for {} (complete:  {} ) roi which lies outside of its provided roi {} ".format(type,
                                                                        complete_chunk_roi, provider_spec_type[type].roi)
        offset = np.array(initial_offset)
        covered_final_roi = False
        while not covered_final_roi:
            # create a copy of the requested batch spec
            chunk_request = copy.deepcopy(self.request)
            max_strides = []
            # change size and offset of the batch spec
            for chunk_spec_type, chunk_request_type, request_type, provider_spec_type in zip([self.chunk_spec_template.volume_specs,
                                                                                              self.chunk_spec_template.points_specs],
                                                                        [chunk_request.volume_specs, chunk_request.points_specs],
                                                                        [self.request.volume_specs, self.request.points_specs],
                                                                        [self.spec.volume_specs, self.spec.points_specs]):
                for type, spec in chunk_spec_type.items():
                    roi = spec.roi
                    chunk_request_type[type].roi = roi + Coordinate(offset)
                    max_stride = np.zeros([roi.dims()])
                    for dim in range(roi.dims()):
                        if chunk_request_type[type].roi.get_end()[dim] <= (request_type[type].roi.get_end()[dim]-chunk_spec_type[type].roi.get_shape()[dim]):
                            max_stride[dim] = max((request_type[type].roi.get_begin()[dim] - chunk_request_type[type].roi.get_begin()[dim]), chunk_spec_type[type].roi.get_shape()[dim])
                        else:
                            if chunk_request_type[type].roi.get_end()[dim] < request_type[type].roi.get_end()[dim]:
                                max_stride[dim] = np.min((chunk_spec_type[type].roi.get_shape()[dim],
                                                          provider_spec_type[type].roi.get_end()[dim] -
                                                          chunk_request_type[type].roi.get_end()[dim]))
                            else:
                                max_stride[dim] = np.min((final_offset[dim]-offset[dim],
                                                          provider_spec_type[type].roi.get_end()[dim] - chunk_request_type[type].roi.get_end()[dim]))
                    max_strides.append(max_stride)

            stride = np.min(max_strides, axis=0)

            logger.debug("requesting chunk " + str(chunk_request))
            self.requests.put(chunk_request)

            if (offset >= final_offset).all():
                covered_final_roi = True
            for d in range(self.dims):
                if offset[d] >= final_offset[d]:
                    if d == self.dims - 1:
                        break
                    offset[d] = initial_offset[d]
                else:
                    offset[d] += stride[d]
                    break

        self.num_requests = int(self.requests.qsize())
