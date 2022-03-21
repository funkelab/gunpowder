import math
import logging
from random import random, randint, choices, seed
import itertools

import numpy as np
from scipy.spatial import cKDTree
from skimage.transform import integral_image, integrate
from gunpowder.batch_request import BatchRequest
from gunpowder.coordinate import Coordinate
from gunpowder.roi import Roi
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)


class RandomLocation(BatchFilter):
    """Choses a batch at a random location in the bounding box of the upstream
    provider.

    The random location is chosen such that the batch request ROI lies entirely
    inside the provider's ROI.

    If ``min_masked`` and ``mask`` are set, only batches are returned that have
    at least the given ratio of masked-in voxels. This is in general faster
    than using the :class:`Reject` node, at the expense of storing an integral
    array of the complete mask.

    If ``ensure_nonempty`` is set to a :class:`GraphKey`, only batches are
    returned that have at least one point of this point collection within the
    requested ROI.

    Additional tests for randomly picked locations can be implemented by
    subclassing and overwriting of :func:`accepts`. This method takes the
    randomly shifted request that meets all previous criteria (like
    ``min_masked`` and ``ensure_nonempty``) and should return ``True`` if the
    request is acceptable.

    Args:

        min_masked (``float``, optional):

            If non-zero, require that the random sample contains at least that
            ratio of masked-in voxels.

        mask (:class:`ArrayKey`, optional):

            The array to use for mask checks.

        ensure_nonempty (:class:`GraphKey`, optional):

            Ensures that when finding a random location, a request for
            ``ensure_nonempty`` will contain at least one point.

        p_nonempty (``float``, optional):

            If ``ensure_nonempty`` is set, it defines the probability that a
            request for ``ensure_nonempty`` will contain at least one point.
            Default value is 1.0.

        ensure_centered (``bool``, optional):

            if ``ensure_nonempty`` is set, ``ensure_centered`` guarantees
            that the center voxel of the roi contains a point.

        point_balance_radius (``int``):

            if ``ensure_nonempty`` is set, ``point_balance_radius`` defines
            a radius s.t. for every point `p` in ``ensure_nonempty``, the
            probability of picking p is inversely related to the number of
            other points within a distance of ``point_balance_radius`` to p.
            This helps avoid oversampling of dense regions of the graph, and
            undersampling of sparse regions.

        random_shift_key (``ArrayKey`` optional):

            if ``random_shift_key`` is not None, this node will populate
            that key with a nonspatial array containing the random shift
            used for each request. This can be useful for snapshot iterations
            if you want to figure out where that snapshot came from.
    """

    def __init__(
        self,
        min_masked=0,
        mask=None,
        ensure_nonempty=None,
        p_nonempty=1.0,
        ensure_centered=None,
        point_balance_radius=1,
        random_shift_key=None,
    ):

        self.min_masked = min_masked
        self.mask = mask
        self.mask_spec = None
        self.mask_integral = None
        self.ensure_nonempty = ensure_nonempty
        self.points = None
        self.p_nonempty = p_nonempty
        self.upstream_spec = None
        self.random_shift = None
        self.ensure_centered = ensure_centered
        self.point_balance_radius = point_balance_radius
        self.random_shift_key = random_shift_key

    def setup(self):

        upstream = self.get_upstream_provider()
        self.upstream_spec = upstream.spec

        if self.mask and self.min_masked > 0:

            assert self.mask in self.upstream_spec, (
                "Upstream provider does not have %s"%self.mask)
            self.mask_spec = self.upstream_spec.array_specs[self.mask]

            logger.info("requesting complete mask...")

            mask_request = BatchRequest({self.mask: self.mask_spec})
            mask_batch = upstream.request_batch(mask_request)

            logger.info("allocating mask integral array...")

            mask_data = mask_batch.arrays[self.mask].data
            mask_integral_dtype = np.uint64
            logger.debug("mask size is %s", mask_data.size)
            if mask_data.size < 2**32:
                mask_integral_dtype = np.uint32
            if mask_data.size < 2**16:
                mask_integral_dtype = np.uint16
            logger.debug("chose %s as integral array dtype", mask_integral_dtype)

            self.mask_integral = np.array(mask_data > 0, dtype=mask_integral_dtype)
            self.mask_integral = integral_image(self.mask_integral).astype(mask_integral_dtype)

        if self.ensure_nonempty:

            assert self.ensure_nonempty in self.upstream_spec, (
                "Upstream provider does not have %s"%self.ensure_nonempty)
            graph_spec = self.upstream_spec.graph_specs[self.ensure_nonempty]


            logger.info("requesting all %s points...", self.ensure_nonempty)

            nonempty_request = BatchRequest({self.ensure_nonempty: graph_spec})
            nonempty_batch = upstream.request_batch(nonempty_request)

            self.points = cKDTree(
                [p.location for p in nonempty_batch[self.ensure_nonempty].nodes]
            )

            point_counts = self.points.query_ball_point(
                [p.location for p in nonempty_batch[self.ensure_nonempty].nodes],
                r=self.point_balance_radius,
            )
            weights = [1 / len(point_count) for point_count in point_counts]
            self.cumulative_weights = list(itertools.accumulate(weights))

            logger.debug("retrieved %d points", len(self.points.data))

        # clear bounding boxes of all provided arrays and points --
        # RandomLocation does not have limits (offsets are ignored)
        for key, spec in self.spec.items():
            if spec.roi is not None:
                spec.roi.shape = Coordinate((None,)*spec.roi.dims)
                self.updates(key, spec)

        # provide randomness if asked for
        if self.random_shift_key is not None:
            self.provides(self.random_shift_key, ArraySpec(nonspatial=True))

    def prepare(self, request):
        seed(request.random_seed)

        logger.debug("request: %s", request.array_specs)
        logger.debug("my spec: %s", self.spec)

        if request.array_specs.keys():
            lcm_voxel_size = self.spec.get_lcm_voxel_size(
                request.array_specs.keys())
        else:
            lcm_voxel_size = Coordinate((1,)*request.get_total_roi().dims)

        shift_roi = self.__get_possible_shifts(request, lcm_voxel_size)

        if request.array_specs.keys():

            shift_roi = shift_roi.snap_to_grid(lcm_voxel_size, mode='shrink')
            lcm_shift_roi = shift_roi/lcm_voxel_size
            logger.debug(
                "restricting random locations to multiples of voxel size %s",
                lcm_voxel_size)

        else:

            lcm_shift_roi = shift_roi

        assert not lcm_shift_roi.unbounded, (
            "Can not pick a random location, intersection of upstream ROIs is "
            "unbounded.")
        assert not lcm_shift_roi.empty, (
            "Can not satisfy batch request, no location covers all requested "
            "ROIs.")

        random_shift = self.__select_random_shift(
            request,
            lcm_shift_roi,
            lcm_voxel_size)

        self.random_shift = random_shift
        self.__shift_request(request, random_shift)

        return request

    def process(self, batch, request):

        if self.random_shift_key is not None:
            batch[self.random_shift_key] = Array(
                np.array(self.random_shift),
                ArraySpec(nonspatial=True),
            )

        # reset ROIs to request
        for (array_key, spec) in request.array_specs.items():
            batch.arrays[array_key].spec.roi = spec.roi
        for (graph_key, spec) in request.graph_specs.items():
            batch.graphs[graph_key].spec.roi = spec.roi

        # change shift point locations to lie within roi
        for graph_key in request.graph_specs.keys():
            batch.graphs[graph_key].shift(-self.random_shift)

    def accepts(self, request):
        '''Should return True if the randomly chosen location is acceptable
        (besided meeting other criteria like ``min_masked`` and/or
        ``ensure_nonempty``). Subclasses can overwrite this method to implement
        additional tests for acceptable locations.'''

        return True

    def __get_possible_shifts(self, request, voxel_size):

        total_shift_roi = None

        for key, spec in request.items():

            if spec.roi is None:
                continue

            request_roi = spec.roi
            provided_roi = self.upstream_spec[key].roi

            shift_roi = provided_roi.shift(
                -request_roi.begin
            ).grow(
                (0,)*request_roi.dims,
                -(request_roi.shape - voxel_size)
            )

            if total_shift_roi is None:
                total_shift_roi = shift_roi
            else:
                if shift_roi != total_shift_roi:
                    total_shift_roi = total_shift_roi.intersect(shift_roi)

        logger.debug("valid shifts for request in " + str(total_shift_roi))

        return total_shift_roi

    def __select_random_shift(self, request, lcm_shift_roi, lcm_voxel_size):

        ensure_points = (
            self.ensure_nonempty is not None
            and
            random() <= self.p_nonempty)

        while True:

            if ensure_points:
                random_shift = self.__select_random_location_with_points(
                    request,
                    lcm_shift_roi,
                    lcm_voxel_size)
            else:
                random_shift = self.__select_random_location(
                    lcm_shift_roi,
                    lcm_voxel_size)

            logger.debug("random shift: " + str(random_shift))

            if not self.__is_min_masked(random_shift, request):
                logger.debug(
                    "random location does not meet 'min_masked' criterium")
                continue

            if not self.__accepts(random_shift, request):
                logger.debug(
                    "random location does not meet user-provided criterium")
                continue

            return random_shift

    def __is_min_masked(self, random_shift, request):

        if not self.mask or self.min_masked == 0:
            return True

        # get randomly chosen mask ROI
        request_mask_roi = request.array_specs[self.mask].roi
        request_mask_roi = request_mask_roi.shift(random_shift)

        # get coordinates inside mask array
        mask_voxel_size = self.spec[self.mask].voxel_size
        request_mask_roi_in_array = request_mask_roi/mask_voxel_size
        request_mask_roi_in_array -= self.mask_spec.roi.offset/mask_voxel_size

        # get number of masked-in voxels
        num_masked_in = integrate(
            self.mask_integral,
            [request_mask_roi_in_array.begin],
            [request_mask_roi_in_array.end-(1,)*self.mask_integral.ndim]
        )[0]

        mask_ratio = float(num_masked_in)/request_mask_roi_in_array.size()
        logger.debug("mask ratio is %f", mask_ratio)

        return mask_ratio >= self.min_masked

    def __accepts(self, random_shift, request):

        # create a shifted copy of the request
        shifted_request = request.copy()
        self.__shift_request(shifted_request, random_shift)

        return self.accepts(shifted_request)

    def __shift_request(self, request, shift):

        # shift request ROIs
        for specs_type in [request.array_specs, request.graph_specs]:
            for (key, spec) in specs_type.items():
                if spec.roi is None:
                    continue
                roi = spec.roi.shift(shift)
                specs_type[key].roi = roi

    def __select_random_location_with_points(
            self,
            request,
            lcm_shift_roi,
            lcm_voxel_size):

        request_points = request.graph_specs.get(self.ensure_nonempty)
        if request_points is None:
            total_roi = request.get_total_roi()
            logger.warning(
                f"Requesting non empty {self.ensure_nonempty}, however {self.ensure_nonempty} "
                f"has not been requested. Falling back on using the total roi of the "
                f"request {total_roi} for {self.ensure_nonempty}."
            )
            request_points_roi = total_roi
        else:
            request_points_roi = request_points.roi

        while True:

            # How to pick shifts that ensure that a randomly chosen point is
            # contained in the request ROI:
            #
            #
            # request          point
            # [---------)      .
            # 0        +10     17
            #
            #         least shifted to contain point
            #         [---------)
            #         8        +10
            #         ==
            #         point-request.begin-request.shape+1
            #
            #                  most shifted to contain point:
            #                  [---------)
            #                  17       +10
            #                  ==
            #                  point-request.begin
            #
            #         all possible shifts
            #         [---------)
            #         8        +10
            #         ==
            #         point-request.begin-request.shape+1
            #                   ==
            #                   request.shape

            # pick a random point
            point = choices(self.points.data, cum_weights=self.cumulative_weights)[0]

            logger.debug("select random point at %s", point)

            # get the lcm voxel that contains this point
            lcm_location = Coordinate(point/lcm_voxel_size)
            logger.debug(
                "belongs to lcm voxel %s",
                lcm_location)

            # get the request ROI's shape in lcm
            lcm_roi_begin = request_points_roi.begin/lcm_voxel_size
            lcm_roi_shape = request_points_roi.shape/lcm_voxel_size
            logger.debug("Point request ROI: %s", request_points_roi)
            logger.debug("Point request lcm ROI shape: %s", lcm_roi_shape)

            # get all possible starting points of lcm_roi_shape that contain
            # lcm_location
            if self.ensure_centered:
                lcm_shift_roi_begin = (
                    lcm_location
                    - lcm_roi_begin
                    - lcm_roi_shape / 2
                    + Coordinate((1,) * len(lcm_location))
                )
                lcm_shift_roi_shape = Coordinate((1,) * len(lcm_location))
            else:
                lcm_shift_roi_begin = (
                    lcm_location - lcm_roi_begin - lcm_roi_shape +
                    Coordinate((1,)*len(lcm_location))
                )
                lcm_shift_roi_shape = lcm_roi_shape
            lcm_point_shift_roi = Roi(lcm_shift_roi_begin, lcm_shift_roi_shape)

            logger.debug("lcm point shift roi: %s", lcm_point_shift_roi)

            # intersect with total shift ROI
            if not lcm_point_shift_roi.intersects(lcm_shift_roi):
                logger.debug(
                    "reject random shift, random point %s shift ROI %s does "
                    "not intersect total shift ROI %s", point,
                    lcm_point_shift_roi, lcm_shift_roi)
                continue
            lcm_point_shift_roi = lcm_point_shift_roi.intersect(lcm_shift_roi)

            # select a random shift from all possible shifts
            random_shift = self.__select_random_location(
                lcm_point_shift_roi,
                lcm_voxel_size)
            logger.debug("random shift: %s", random_shift)

            # count all points inside the shifted ROI
            points = self.__get_points_in_roi(
                request_points_roi.shift(random_shift))
            assert point in points, (
                "Requested batch to contain point %s, but got points "
                "%s"%(point, points))
            num_points = len(points)

            return random_shift

    def __select_random_location(self, lcm_shift_roi, lcm_voxel_size):

        # select a random point inside ROI
        random_shift = Coordinate(
            randint(begin, end - 1)
            for begin, end in zip(lcm_shift_roi.begin, lcm_shift_roi.end))

        random_shift *= lcm_voxel_size

        return random_shift

    def __get_points_in_roi(self, roi):

        points = []

        center = roi.center
        radius = math.ceil(float(max(roi.shape))/2)
        candidates = self.points.query_ball_point(center, radius, p=np.inf)

        for i in candidates:
            if roi.contains(self.points.data[i]):
                points.append(self.points.data[i])

        return np.array(points)
