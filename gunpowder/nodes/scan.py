import logging
import multiprocessing
import numpy as np
from gunpowder.array import Array
from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.points import Points
from gunpowder.producer_pool import ProducerPool
from gunpowder.roi import Roi
from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class Scan(BatchFilter):
    '''Iteratively requests batches of size ``reference`` from upstream
    providers in a scanning fashion, until all requested ROIs are covered. If
    the batch request to this node is empty, it will scan the complete upstream
    ROIs (and return nothing). Otherwise, it scans only the requested ROIs and
    returns a batch assembled of the smaller requests. In either case, the
    upstream requests will be contained in the downstream requested ROI or
    upstream ROIs.

    See also :class:`Hdf5Write`.

    Args:

        reference (:class:`BatchRequest`):

            A reference :class:`BatchRequest`. This request will be shifted in
            a scanning fashion over the upstream ROIs of the requested arrays
            or points.

        num_workers (``int``, optional):

            If set to >1, upstream requests are made in parallel with that
            number of workers.

        cache_size (``int``, optional):

            If multiple workers are used, how many batches to hold at most.
    '''

    def __init__(self, reference, num_workers=1, cache_size=50):

        self.reference = reference.copy()
        self.num_workers = num_workers
        self.cache_size = cache_size
        self.workers = None
        if num_workers > 1:
            self.request_queue = multiprocessing.Queue(maxsize=0)
        self.batch = None

    def setup(self):

        if self.num_workers > 1:
            self.workers = ProducerPool(
                [self.__worker_get_chunk for _ in range(self.num_workers)],
                queue_size=self.cache_size)
            self.workers.start()

    def teardown(self):

        if self.num_workers > 1:
            self.workers.stop()

    def provide(self, request):

        empty_request = (len(request) == 0)
        if empty_request:
            scan_spec = self.spec
        else:
            scan_spec = request

        stride = self.__get_stride()
        shift_roi = self.__get_shift_roi(scan_spec)

        shifts = self.__enumerate_shifts(shift_roi, stride)
        num_chunks = len(shifts)

        logger.info("scanning over %d chunks", num_chunks)

        # the batch to return
        self.batch = Batch()

        if self.num_workers > 1:

            for shift in shifts:
                shifted_reference = self.__shift_request(self.reference, shift)
                self.request_queue.put(shifted_reference)

            for i in range(num_chunks):

                chunk = self.workers.get()

                if not empty_request:
                    self.__add_to_batch(request, chunk)

                logger.info("processed chunk %d/%d", i + 1, num_chunks)

        else:

            for i, shift in enumerate(shifts):

                shifted_reference = self.__shift_request(self.reference, shift)
                chunk = self.__get_chunk(shifted_reference)

                if not empty_request:
                    self.__add_to_batch(request, chunk)

                logger.info("processed chunk %d/%d", i + 1, num_chunks)

        batch = self.batch
        self.batch = None

        logger.debug("returning batch %s", batch)

        return batch

    def __get_stride(self):
        '''Get the maximal amount by which ``reference`` can be moved, such
        that it tiles the space.'''

        stride = None

        # get the least common multiple of all voxel sizes, we have to stride
        # at least that far
        lcm_voxel_size = self.spec.get_lcm_voxel_size(
            self.reference.array_specs.keys())

        # that's just the minimal size in each dimension
        for key, reference_spec in self.reference.items():

            shape = reference_spec.roi.get_shape()

            for d in range(len(lcm_voxel_size)):
                assert shape[d] >= lcm_voxel_size[d], ("Shape of reference "
                                                       "ROI %s for %s is "
                                                       "smaller than least "
                                                       "common multiple of "
                                                       "voxel size "
                                                       "%s"%(reference_spec.roi,
                                                             key,
                                                             lcm_voxel_size))

            if stride is None:
                stride = shape
            else:
                stride = Coordinate((
                    min(a, b)
                    for a, b in zip(stride, shape)))

        return stride

    def __get_shift_roi(self, spec):
        '''Get the minimal and maximal shift (as a ROI) to apply to
        ``self.reference``, such that it is still fully contained in ``spec``.
        '''

        total_shift_roi = None

        # get individual shift ROIs and intersect them
        for key, reference_spec in self.reference.items():

            logger.debug(
                "getting shift roi for %s with spec %s",
                key,
                reference_spec)

            if key not in spec:
                logger.debug("skipping, %s not in upstream spec", key)
                continue
            if spec[key].roi is None:
                logger.debug("skipping, %s has not ROI", key)
                continue

            logger.debug("upstream ROI is %s", spec[key].roi)

            for r, s in zip(
                    reference_spec.roi.get_shape(),
                    spec[key].roi.get_shape()):
                assert r <= s, (
                    "reference %s with ROI %s does not fit into provided "
                    "upstream %s"%(key, reference_spec.roi, spec[key].roi))

            # we have a reference ROI
            #
            #    [--------) [9]
            #    3        12
            #
            # and a spec ROI
            #
            #                 [---------------) [16]
            #                 16              32
            #
            # min and max shifts of reference are
            #
            #                 [--------) [9]
            #                 16       25
            #                        [--------) [9]
            #                        23       32
            #
            # therefore, all possible ways to shift the reference such that it
            # is contained in the spec are at least 16-3=13 and at most 23-3=20
            # (inclusive)
            #
            #              [-------) [8]
            #              13      21
            #
            # 1. the starting point is beginning of spec - beginning of reference
            # 2. the length is length of spec - length of reference + 1

            # 1. get the starting point of the shift ROI
            shift_begin = (
                spec[key].roi.get_begin() -
                reference_spec.roi.get_begin())

            # 2. get the shape of the shift ROI
            shift_shape = (
                spec[key].roi.get_shape() -
                reference_spec.roi.get_shape() +
                (1,)*reference_spec.roi.dims())

            # create a ROI...
            shift_roi = Roi(shift_begin, shift_shape)

            logger.debug("shift ROI for %s is %s", key, shift_roi)

            # ...and intersect it with previous shift ROIs
            if total_shift_roi is None:
                total_shift_roi = shift_roi
            else:
                total_shift_roi = total_shift_roi.intersect(shift_roi)
                if total_shift_roi.empty():
                    raise RuntimeError("There is no location where the ROIs "
                                       "the reference %s are contained in the "
                                       "request/upstream ROIs "
                                       "%s."%(self.reference, spec))

            logger.debug("intersected with total shift ROI this yields %s",
                    total_shift_roi)

        if total_shift_roi is None:
            raise RuntimeError("None of the upstream ROIs are bounded (all "
                               "ROIs are None). Scan needs at least one "
                               "bounded upstream ROI.")

        return total_shift_roi

    def __enumerate_shifts(self, shift_roi, stride):
        '''Produces a sequence of shift coordinates starting at the beginning
        of ``shift_roi``, progressing with ``stride``. The maximum shift
        coordinate in any dimension will be the last point inside the shift roi
        in this dimension.'''

        min_shift = shift_roi.get_offset()
        max_shift = Coordinate(m - 1 for m in shift_roi.get_end())

        shift = np.array(min_shift)
        shifts = []

        dims = len(min_shift)

        logger.debug(
            "enumerating possible shifts of %s in %s", stride, shift_roi)

        while True:

            logger.debug("adding %s", shift)
            shifts.append(Coordinate(shift))

            if (shift == max_shift).all():
                break

            # count up dimensions
            for d in range(dims):

                if shift[d] >= max_shift[d]:
                    if d == dims - 1:
                        break
                    shift[d] = min_shift[d]
                else:
                    shift[d] += stride[d]
                    # snap to last possible shift, don't overshoot
                    if shift[d] > max_shift[d]:
                        shift[d] = max_shift[d]
                    break

        return shifts

    def __shift_request(self, request, shift):

        shifted = request.copy()
        for _, spec in shifted.items():
            spec.roi = spec.roi.shift(shift)

        return shifted

    def __worker_get_chunk(self):

        request = self.request_queue.get()
        return self.__get_chunk(request)

    def __get_chunk(self, request):

        return self.get_upstream_provider().request_batch(request)

    def __add_to_batch(self, spec, chunk):

        if self.batch.get_total_roi() is None:
            self.batch = self.__setup_batch(spec, chunk)

        for (array_key, array) in chunk.arrays.items():
            if array_key not in spec:
                continue
            self.__fill(self.batch.arrays[array_key].data, array.data,
                        spec.array_specs[array_key].roi, array.spec.roi,
                        self.spec[array_key].voxel_size)

        for (points_key, points) in chunk.points.items():
            if points_key not in spec:
                continue
            self.__fill_points(self.batch.points[points_key].data, points.data,
                               spec.points_specs[points_key].roi, points.roi)

    def __setup_batch(self, batch_spec, chunk):
        '''Allocate a batch matching the sizes of ``batch_spec``, using
        ``chunk`` as template.'''

        batch = Batch()

        for (array_key, spec) in batch_spec.array_specs.items():
            roi = spec.roi
            voxel_size = self.spec[array_key].voxel_size

            # get the 'non-spatial' shape of the chunk-batch
            # and append the shape of the request to it
            array = chunk.arrays[array_key]
            shape = array.data.shape[:-roi.dims()]
            shape += (roi.get_shape() // voxel_size)

            spec = self.spec[array_key].copy()
            spec.roi = roi
            logger.info("allocating array of shape %s for %s", shape, array_key)
            batch.arrays[array_key] = Array(data=np.zeros(shape),
                                                spec=spec)

        for (points_key, spec) in batch_spec.points_specs.items():
            roi = spec.roi
            spec = self.spec[points_key].copy()
            spec.roi = roi
            batch.points[points_key] = Points(data={}, spec=spec)

        logger.debug("setup batch to fill %s", batch)

        return batch

    def __fill(self, a, b, roi_a, roi_b, voxel_size):
        logger.debug("filling " + str(roi_b) + " into " + str(roi_a))

        roi_a = roi_a // voxel_size
        roi_b = roi_b // voxel_size

        common_roi = roi_a.intersect(roi_b)
        if common_roi.empty():
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
