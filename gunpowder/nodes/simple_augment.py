import logging
import random
import itertools

import numpy as np

from .batch_filter import BatchFilter
from gunpowder.coordinate import Coordinate

logger = logging.getLogger(__name__)


class SimpleAugment(BatchFilter):
    """Randomly mirror and transpose all :class:`Arrays<Array>` and
    :class:`Graph` in a batch.

    Args:

        mirror_only (``list`` of ``int``, optional):

            If set, only mirror between the given axes. This is useful to
            exclude channels that have a set direction, like time.

        transpose_only (``list`` of ``int``, optional):

            If set, only transpose between the given axes. This is useful to
            limit the transpose to axes with the same resolution or to exclude
            non-spatial dimensions.

        mirror_probs (``list`` of ``float``, optional):

            If set, provides the probability for mirroring given axes. Default
            is 0.5 per axis. If given, must be given for every axis. i.e.
            [0,1,0] for 100% chance of mirroring axis 1 an no others.

        transpose_probs (``dict`` of ``tuple`` -> ``float``
        or ``list`` of ``float``, optional):

            The probability of transposing. If None, each transpose is equally
            likely.
            Can also be a dictionary of for ``tuple`` -> ``float``. For example
            {(0,1,2):0.5, (1,0,2):0.5} to define a 50% chance of transposing axes
            0 and 1. Note that if a provided option violates the `transpose_only`
            arg it will be dropped and remaining options will be reweighted.
            Can also be provided as a list of ``float``. i.e. [0.3, 0.5, 0.7].
            This will automatically generate a list of possible permutations
            and attempt to weight them appropriately. A weight of 0 means
            this axis will never be transposed, a weight of 1 means this axis
            will always be transposed.
    """

    def __init__(
        self,
        mirror_only=None,
        transpose_only=None,
        mirror_probs=None,
        transpose_probs=None,
    ):

        self.mirror_only = mirror_only
        self.mirror_probs = mirror_probs
        self.transpose_only = transpose_only
        self.transpose_probs = transpose_probs
        self.mirror_mask = None
        self.dims = None
        self.transpose_dims = None

    def setup(self):

        self.dims = self.spec.get_total_roi().dims()

        # mirror_mask and transpose_dims refer to the indices of the spatial
        # dimensions only, starting counting at 0 for the first spatial
        # dimension

        if self.mirror_only is None:
            self.mirror_mask = [True] * self.dims
        else:
            self.mirror_mask = [d in self.mirror_only for d in range(self.dims)]
        if self.mirror_probs is None:
            self.mirror_probs = [0.5] * self.dims

        if self.transpose_only is None:
            self.transpose_dims = list(range(self.dims))
        else:
            self.transpose_dims = self.transpose_only
        if self.transpose_probs is None:
            self.permutation_dict = None
        elif isinstance(self.transpose_probs, list):
            self.permutation_dict = {}
            for permutation in itertools.permutations(range(self.dims), self.dims):
                total_prob = 1
                for i, j, p in zip(range(self.dims), permutation, self.transpose_probs):
                    if i not in self.transpose_dims and i != j:
                        total_prob = 0
                    else:
                        total_prob *= (1 - p) if i == j else p
                if total_prob > 0:
                    self.permutation_dict[permutation] = total_prob
        elif isinstance(self.transpose_probs, dict):
            self.permutation_dict = {}
            for k, v in self.transpose_probs.items():
                valid = True
                for i, j in enumerate(k):
                    if i not in self.transpose_only and i != j:
                        valid = False
                if valid:
                    self.permutation_dict[k] = v

    def prepare(self, request):
        random.seed(request.random_seed)

        self.mirror = [
            random.random() < self.mirror_probs[d] if self.mirror_mask[d] else 0
            for d in range(self.dims)
        ]

        if self.permutation_dict is not None:
            t = random.choices(
                list(self.permutation_dict.keys()),
                weights=list(self.permutation_dict.values()),
                k=1,
            )[0]
        else:
            t = random.sample(self.transpose_dims, k=len(self.transpose_dims))

        self.transpose = list(range(self.dims))
        for o, n in zip(self.transpose_dims, t):
            self.transpose[o] = n

        logger.debug("mirror = %s", self.mirror)
        logger.debug("transpose = %s", self.transpose)

        reverse_transpose = [0] * self.dims
        for d in range(self.dims):
            reverse_transpose[self.transpose[d]] = d

        logger.debug("downstream request = %s", request)

        self.__transpose_request(request, reverse_transpose)
        self.__mirror_request(request, self.mirror)

        logger.debug("upstream request = %s", request)

        return request

    def process(self, batch, request):

        # mirror and transpose ROIs of arrays & points in batch
        total_roi = batch.get_total_roi().copy()
        requested_keys = request.array_specs.keys()
        lcm_voxel_size = self.spec.get_lcm_voxel_size(requested_keys)

        for collection_type in [batch.arrays, batch.graphs]:
            for (key, collector) in collection_type.items():
                if key not in request:
                    continue
                if collector.spec.roi is None:
                    continue
                logger.debug("total ROI = %s", batch.get_total_roi())
                logger.debug("upstream %s ROI = %s", key, collector.spec.roi)
                self.__mirror_roi(collector.spec.roi, total_roi, self.mirror)
                logger.debug("mirrored %s ROI = %s", key, collector.spec.roi)
                self.__transpose_roi(
                    collector.spec.roi, total_roi, self.transpose, lcm_voxel_size
                )
                logger.debug("transposed %s ROI = %s", key, collector.spec.roi)

        mirror = tuple(slice(None, None, -1 if m else 1) for m in self.mirror)
        # arrays
        for (array_key, array) in batch.arrays.items():

            if array_key not in request:
                continue

            if array.spec.nonspatial:
                continue

            num_channels = len(array.data.shape) - self.dims
            channel_slices = (slice(None, None),) * num_channels

            array.data = array.data[channel_slices + mirror]

            transpose = [t + num_channels for t in self.transpose]
            array.data = array.data = array.data.transpose(
                list(range(num_channels)) + transpose
            )

        # graphs
        total_roi_offset = total_roi.get_offset()
        total_roi_center = total_roi.get_center()
        if lcm_voxel_size is not None:
            nearest_voxel_shift = Coordinate(
                (d % v) for d, v in zip(total_roi_center, lcm_voxel_size)
            )
            total_roi_center = total_roi_center - nearest_voxel_shift
        total_roi_end = total_roi.get_end()
        logger.debug("augmenting in %s and center %s", total_roi, total_roi_center)

        for (graph_key, graph) in batch.graphs.items():

            if graph_key not in request:
                continue

            logger.debug("converting nodes in graph %s", graph_key)
            for node in list(graph.nodes):

                logger.debug("old location: %s, %s", node.id, node.location)

                # mirror
                location_in_total_offset = np.asarray(node.location) - total_roi_offset
                node.location = np.asarray(
                    [
                        total_roi_end[dim] - location_in_total_offset[dim]
                        if m
                        else node.location[dim]
                        for dim, m in enumerate(self.mirror)
                    ],
                    dtype=graph.spec.dtype,
                )

                logger.debug("after mirror: %s, %s", node.id, node.location)

                # transpose
                location_in_total_center = np.asarray(node.location) - total_roi_center

                if self.transpose != list(range(self.dims)):
                    for d in range(self.dims):
                        node.location[d] = (
                            location_in_total_center[self.transpose[d]]
                            + total_roi_center[d]
                        )

                logger.debug("after transpose: %s, %s", node.id, node.location)

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
        requested_keys = request.array_specs.keys()
        lcm_voxel_size = self.spec.get_lcm_voxel_size(requested_keys)
        for key, spec in request.items():
            if spec.roi is not None:
                self.__transpose_roi(spec.roi, total_roi, transpose, lcm_voxel_size)

    def __mirror_roi(self, roi, total_roi, mirror):

        total_roi_offset = total_roi.get_offset()
        total_roi_shape = total_roi.get_shape()

        roi_offset = roi.get_offset()
        roi_shape = roi.get_shape()

        roi_in_total_offset = roi_offset - total_roi_offset
        end_of_roi_in_total = roi_in_total_offset + roi_shape
        roi_in_total_offset_mirrored = total_roi_shape - end_of_roi_in_total
        roi_offset = Coordinate(
            total_roi_offset[d] + roi_in_total_offset_mirrored[d]
            if mirror[d]
            else roi_offset[d]
            for d in range(self.dims)
        )

        roi.set_offset(roi_offset)

    def __transpose_roi(self, roi, total_roi, transpose, lcm_voxel_size):

        logger.debug("original roi = %s", roi)

        center = total_roi.get_center()
        if lcm_voxel_size is not None:
            nearest_voxel_shift = Coordinate(
                (d % v) for d, v in zip(center, lcm_voxel_size)
            )
            center = center - nearest_voxel_shift
        logger.debug("center = %s", center)

        # Get distance from center, then transpose
        dist_to_center = center - roi.get_offset()
        dist_to_center = Coordinate(
            dist_to_center[transpose[d]] for d in range(self.dims)
        )
        logger.debug("dist_to_center = %s", dist_to_center)

        # Using the tranposed distance to center, get the correct offset.
        new_offset = center - dist_to_center
        logger.debug("new_offset = %s", new_offset)

        shape = tuple(roi.get_shape()[transpose[d]] for d in range(self.dims))
        roi.set_offset(new_offset)
        roi.set_shape(shape)
        logger.debug("tranposed roi = %s", roi)
