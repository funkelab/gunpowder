import numpy as np

from daisy.persistence import MongoDbGraphProvider

from gunpowder.batch import Batch
from gunpowder.nodes.batch_provider import BatchProvider
from gunpowder.graph import Graph, GraphKey
from gunpowder.graph_spec import GraphSpec
from gunpowder.roi import Roi
from gunpowder.coordinate import Coordinate
from gunpowder.profiling import Timing

import networkx as nx

import logging
from typing import Tuple, List, Optional, Union

logger = logging.getLogger(__name__)

unbounded = Roi(Coordinate([None, None, None]), Coordinate([None, None, None]))


class DaisyGraphProvider(BatchProvider):
    """
    See documentation for mongo graph provider at
    https://github.com/funkelab/daisy/blob/0.3-dev/daisy/persistence/mongodb_graph_provider.py#L17
    """

    def __init__(
        self,
        dbname: str,
        url: str,
        points: List[GraphKey],
        points_specs: Optional[Union[GraphSpec, List[GraphSpec]]] = None,
        directed: bool = False,
        total_roi: Roi = None,
        nodes_collection: str = "nodes",
        edges_collection: str = "edges",
        meta_collection: str = "meta",
        endpoint_names: Tuple[str, str] = ("u", "v"),
        position_attribute: str = "position",
        node_attrs: Optional[List[str]] = None,
        edge_attrs: Optional[List[str]] = None,
    ):
        self.points = points
        points_specs = (
            points_specs
            if points_specs is not None
            else GraphSpec(Roi(Coordinate([None] * 3), Coordinate([None] * 3)))
        )
        specs = (
            points_specs
            if isinstance(points_specs, list) and len(points_specs) == len(points)
            else [points_specs] * len(points)
        )
        self.specs = {key: spec for key, spec in zip(points, specs)}

        self.position_attribute = position_attribute
        self.node_attrs = node_attrs
        self.edge_attrs = edge_attrs

        self.graph_provider = MongoDbGraphProvider(
            dbname,
            url,
            mode="r+",
            directed=directed,
            total_roi=None,
            nodes_collection=nodes_collection,
            edges_collection=edges_collection,
            meta_collection=meta_collection,
            endpoint_names=endpoint_names,
            position_attribute=position_attribute,
        )

    def setup(self):
        for key, spec in self.specs.items():
            self.provides(key, spec)

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

        for key, spec in request.items():
            requested_graph = self.graph_provider.get_graph(
                spec.roi,
                edge_inclusion="either",
                node_inclusion="dangling",
                node_attrs=self.node_attrs,
                edge_attrs=self.edge_attrs,
            )
            logger.debug(
                f"got {len(requested_graph.nodes)} nodes and {len(requested_graph.edges)} edges"
            )
            graph = nx.DiGraph()
            for node, attrs in requested_graph.nodes.items():
                loc = attrs.pop(self.position_attribute)
                graph.add_node(node, location=np.array(loc), **attrs)
            graph.add_edges_from(
                (u, v, d)
                for u, nbrs in requested_graph._adj.items()
                for v, d in nbrs.items()
                if u in graph.nodes and v in graph.nodes
            )
            points = Graph.from_nx_graph(graph, spec)
            points.crop(spec.roi)
            batch[key] = points

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
