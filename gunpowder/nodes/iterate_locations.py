import logging
import multiprocessing as mp
from random import randrange

from .batch_filter import BatchFilter
from gunpowder.batch_request import BatchRequest
from gunpowder.coordinate import Coordinate
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec

logger = logging.getLogger(__name__)


class IterateLocations(BatchFilter):
    """Iterates over the nodes in a graph and centers
    batches at their locations. The iteration is thread safe.

    Args:
        graph (:class:`GraphKey`): Key of graph to read nodes from

        roi (:class:`Roi`): Roi within which to read and iterate over nodes.
            Defaults to None, which queries the whole Roi of the upstream graph
            source

        node_id (:class:`ArrayKey`, optional): Nonspatial array key in which to
            store the id of the "current" node in graph.  Default is None, in
            which case no attribute is stored and there is no way to tell which
            node is being considered.

        choose_randomly (bool): If true, choose nodes randomly with
            replacement. Default is false, which loops over the list.
    """

    __global_index = mp.Value("i", -1)
    visited_all = mp.Value("b", False)

    def __init__(self, graph, roi=None, node_id=None, choose_randomly=False):
        self.graph = graph
        self.roi = roi
        self.node_id = node_id
        self.choose_randomly = choose_randomly
        self.nodes = None
        self.coordinates = None
        self.local_index = None
        self.shift = None

    def setup(self):
        upstream = self.get_upstream_provider()
        self.upstream_spec = upstream.spec
        assert self.graph in self.upstream_spec, (
            "Upstream provider does not have graph %s" % self.graph
        )
        query_spec = self.upstream_spec.graph_specs[self.graph].copy()
        if self.roi:
            query_spec.roi = query_spec.roi.intersect(self.roi)
        # TODO: For scalability, scan upstream roi in blocks instead of
        #       storing all nodes in memory
        logger.info("Requesting all %s points in roi %s", self.graph, query_spec.roi)
        upstream_request = BatchRequest({self.graph: query_spec})
        upstream_batch = upstream.request_batch(upstream_request)
        self.nodes = list(upstream_batch[self.graph].nodes)
        self.coordinates = [node.location for node in self.nodes]
        assert (
            len(self.coordinates) > 0
        ), "Graph  %s doesn't have nodes to iterate over in roi %s" % (
            self.graph,
            self.roi,
        )

        # clear bounding boxes of all provided arrays and points
        for key, spec in self.spec.items():
            if spec.roi is not None:
                spec.roi.shape = Coordinate((None,) * spec.roi.dims)
                self.updates(key, spec)
        if self.node_id is not None:
            self.provides(self.node_id, ArraySpec(nonspatial=True))

    def prepare(self, request):
        logger.debug("request: %s", request.array_specs)
        logger.debug("my spec: %s", self.spec)

        lcm_voxel_size = self.spec.get_lcm_voxel_size(request.array_specs.keys())
        if lcm_voxel_size is None:
            ndims = len(self.coordinates[0])
            lcm_voxel_size = Coordinate((1,) * ndims)

        # shift to center
        total_roi = request.get_total_roi()
        request_center = total_roi.shape / 2 + total_roi.offset

        self.shift = self._get_next_shift(request_center, lcm_voxel_size)
        max_tries = 15
        tries = 0
        while not self.__check_shift(request):
            logger.warning(
                "Location %s (shift %s) skipped"
                % (self.coordinates[self.local_index], self.shift)
            )
            assert tries < max_tries, (
                "Unable to find valid shift after %d tries",
                tries,
            )
            self.shift = self._get_next_shift(request_center, lcm_voxel_size)
            tries += 1

        # Set shift for all requests
        for specs_type in [request.array_specs, request.graph_specs]:
            for key, spec in specs_type.items():
                if isinstance(spec, ArraySpec) and spec.nonspatial:
                    continue
                roi = spec.roi.shift(self.shift)
                specs_type[key].roi = roi

        logger.debug(
            "{}'th ({}) shift selected: {}".format(
                self.local_index, self.coordinates[self.local_index], self.shift
            )
        )

    def process(self, batch, request):
        if self.node_id:
            node_id = self.nodes[self.local_index].id
            spec = self.spec[self.node_id].copy()
            batch[self.node_id] = Array([node_id], spec)

        # reset ROIs to request
        for array_key, spec in request.array_specs.items():
            batch.arrays[array_key].spec.roi = spec.roi

        for graph_key, spec in request.graph_specs.items():
            batch.graphs[graph_key].spec.roi = spec.roi

        # change shift point locations to lie within roi
        for graph_key in request.graph_specs.keys():
            batch.graphs[graph_key].shift(-self.shift)

    def _get_next_shift(self, center_shift, voxel_size):
        # gets next coordinate from list
        if self.choose_randomly:
            self.local_index = randrange(len(self.coordinates))
        else:
            with IterateLocations.__global_index.get_lock():
                IterateLocations.__global_index.value += 1
                if IterateLocations.__global_index.value == len(self.coordinates) - 1:
                    logger.info("After this request, all points have been visited")
                    with IterateLocations.visited_all.get_lock():
                        IterateLocations.visited_all.value = True
                if IterateLocations.__global_index.value == len(self.coordinates):
                    logger.warning("Ran out of locations, looping list")
                self.local_index = IterateLocations.__global_index.value % len(
                    self.coordinates
                )
        next_shift = Coordinate(self.coordinates[self.local_index]) - center_shift

        logger.debug("Shift before rounding: %s" % str(next_shift))
        # make sure shift is a multiple of voxel size (round to nearest)
        next_shift = Coordinate(
            [
                int(vs * round(float(shift) / vs))
                for vs, shift in zip(voxel_size, next_shift)
            ]
        )
        logger.debug("Shift after rounding: %s" % str(next_shift))
        return next_shift

    def __check_shift(self, request):
        for key, spec in request.items():
            if isinstance(spec, ArraySpec) and spec.nonspatial:
                continue
            request_roi = spec.roi
            if key in self.upstream_spec:
                provided_roi = self.upstream_spec[key].roi
            else:
                raise Exception("Requested %s, but upstream does not provide it." % key)
            shifted_roi = request_roi.shift(self.shift)
            if not provided_roi.contains(shifted_roi):
                logger.warning(
                    ("Provided roi %s for key %s does notcontain" " shifted roi %s"),
                    provided_roi,
                    key,
                    shifted_roi,
                )
                return False
        return True
