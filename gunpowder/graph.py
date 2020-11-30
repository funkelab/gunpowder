from .graph_spec import GraphSpec
from .roi import Roi
from .freezable import Freezable

import numpy as np
import networkx as nx

from copy import deepcopy
from typing import Dict, Optional, Set, Iterator, Any
import logging
import itertools
import warnings


logger = logging.getLogger(__name__)


class Node(Freezable):
    """
    A stucture representing each node in a Graph.

    Args:

        id (``int``):

            A unique identifier for this Node

        location (``np.ndarray``):

            A numpy array containing a nodes location

        Optional attrs (``dict``, str -> ``Any``):

            A dictionary containing a mapping from attribute to value.
            Used to store any extra attributes associated with the
            Node such as color, size, etc.

        Optional temporary (bool):

            A tag to mark a node as temporary. Some operations such
            as `trim` might make new nodes that are just biproducts
            of viewing the data with a limited scope. These nodes
            are only guaranteed to have an id different from those
            in the same Graph, but may have conflicts if you request
            multiple graphs from the same source with different rois.
    """

    def __init__(
        self,
        id: int,
        location: np.ndarray,
        temporary: bool = False,
        attrs: Optional[Dict[str, Any]] = None,
    ):
        self.__attrs = attrs if attrs is not None else {}
        self.attrs["id"] = id
        self.location = location
        # purpose is to keep track of nodes that were created during
        # processing and do not have a corresponding node in the original source
        self.attrs["temporary"] = temporary
        self.freeze()

    def __getattr__(self, attr):
        if "__" not in attr:
            return self.attrs[attr]
        else:
            return super().__getattr__(attr)

    def __setattr__(self, attr, value):
        if "__" not in attr:
            self.attrs[attr] = value
        else:
            super().__setattr__(attr, value)

    @property
    def location(self):
        location = self.attrs["location"]
        return location

    @location.setter
    def location(self, new_location):
        assert isinstance(new_location, np.ndarray)
        self.attrs["location"] = new_location

    @property
    def id(self):
        return self.attrs["id"]

    @property
    def original_id(self):
        return self.id if not self.temporary else None

    @property
    def temporary(self):
        return self.attrs["temporary"]

    @property
    def attrs(self):
        return self.__attrs

    @property
    def all(self):
        return self.attrs

    @classmethod
    def from_attrs(cls, attrs: Dict[str, Any]):
        node_id = attrs["id"]
        location = attrs["location"]
        temporary = attrs.get("temporary", False)
        return cls(
            id=node_id, location=location, temporary=temporary, attrs=attrs
        )

    def __str__(self):
        return f"Node({self.temporary}) ({self.id}) at ({self.location})"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


class Edge(Freezable):
    """
    A structure representing edges in a graph.

    Args:

        u (``int``)

            The id of the 'u' node of this edge

        v (``int``)

            the id of the `v` node of this edge
    """

    def __init__(self, u: int, v: int, attrs: Optional[Dict[str, Any]] = None):
        self.__u = u
        self.__v = v
        self.__attrs = attrs if attrs is not None else {}
        self.freeze()

    @property
    def u(self):
        return self.__u

    @property
    def v(self):
        return self.__v

    @property
    def all(self):
        return self.__attrs

    def __iter__(self):
        return iter([self.u, self.v])

    def __str__(self):
        return f"({self.u}, {self.v})"

    def __repr__(self):
        return f"({self.u}, {self.v})"

    def __eq__(self, other):
        return self.u == other.u and self.v == other.v

    def __hash__(self):
        return hash((self.u, self.v))

    def directed_eq(self, other):
        return self.u == other.u and self.v == other.v

    def undirected_eq(self, other):
        return set([self.u, self.v]) == set([other.u, other.v])


class Graph(Freezable):
    """A structure containing a list of :class:`Node`, a list of :class:`Edge`,
    and a specification describing the data.

    Args:

        nodes (``iterator``, :class:`Node`):

            An iterator containing Vertices.

        edges (``iterator``, :class:`Edge`):

            An iterator containing Edges.

        spec (:class:`GraphSpec`):

            A spec describing the data.
    """

    def __init__(self, nodes: Iterator[Node], edges: Iterator[Edge], spec: GraphSpec):
        self.__spec = spec
        self.__graph = self.create_graph(nodes, edges)

    @property
    def spec(self):
        return self.__spec

    @spec.setter
    def spec(self, new_spec):
        self.__spec = new_spec

    @property
    def directed(self):
        return (
            self.spec.directed
            if self.spec.directed is not None
            else self.__graph.is_directed()
        )

    def create_graph(self, nodes: Iterator[Node], edges: Iterator[Edge]):
        if self.__spec.directed is None:
            logger.debug(
                "Trying to create a Graph without specifying directionality. Using default Directed!"
            )
            graph = nx.DiGraph()
        elif self.__spec.directed:
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()

        for node in nodes:
            node.location = node.location.astype(self.spec.dtype)

        vs = [(v.id, v.all) for v in nodes]
        graph.add_nodes_from(vs)
        graph.add_edges_from([(e.u, e.v, e.all) for e in edges])
        return graph

    @property
    def nodes(self):
        for node_id, node_attrs in self.__graph.nodes.items():
            v = Node.from_attrs(node_attrs)
            if not np.issubdtype(v.location.dtype, self.spec.dtype):
                raise Exception(
                    f"expected location to have dtype {self.spec.dtype} but it had {v.location.dtype}"
                )
            yield v

    def num_vertices(self):
        return self.__graph.number_of_nodes()

    def num_edges(self):
        return self.__graph.number_of_edges()

    @property
    def edges(self):
        for (u, v), attrs in self.__graph.edges.items():
            yield Edge(u, v, attrs)

    def neighbors(self, node):
        if self.directed:
            for neighbor in self.__graph.successors(node.id):
                yield Node.from_attrs(self.__graph.nodes[neighbor])
            if self.directed:
                for neighbor in self.__graph.predecessors(node.id):
                    yield Node.from_attrs(self.__graph.nodes[neighbor])
        else:
            for neighbor in self.__graph.neighbors(node.id):
                yield Node.from_attrs(self.__graph.nodes[neighbor])

    def __str__(self):
        string = "Vertices:\n"
        for node in self.nodes:
            string += f"{node}\n"
        string += "Edges:\n"
        for edge in self.edges:
            string += f"{edge}\n"
        return string

    def __repr__(self):
        return str(self)

    def node(self, id: int):
        """
        Get node with a specific id
        """
        attrs = self.__graph.nodes[id]
        return Node.from_attrs(attrs)

    def contains(self, node_id: int):
        return node_id in self.__graph.nodes

    def remove_node(self, node: Node, retain_connectivity=False):
        """
        Remove a node.

        retain_connectivity: preserve removed nodes neighboring edges.
        Given graph: a->b->c, removing `b` without retain_connectivity
        would leave us with two connected components, {'a'} and {'b'}.
        removing 'b' with retain_connectivity flag set to True would
        leave us with the graph: a->c, and only one connected component
        {a, c}, thus preserving the connectivity of 'a' and 'c'
        """
        if retain_connectivity:
            predecessors = self.predecessors(node)
            successors = self.successors(node)

            for pred_id in predecessors:
                for succ_id in successors:
                    if pred_id != succ_id:
                        self.add_edge(Edge(pred_id, succ_id))
        self.__graph.remove_node(node.id)

    def add_node(self, node: Node):
        """
        Adds a node to the graph.
        If a node exists with the same id as the node you are adding,
        its attributes will be overwritten.
        """
        node.location = node.location.astype(self.spec.dtype)
        self.__graph.add_node(node.id, **node.all)

    def remove_edge(self, edge: Edge):
        """
        Remove an edge from the graph.
        """
        self.__graph.remove_edge(edge.u, edge.v)

    def add_edge(self, edge: Edge):
        """
        Adds an edge to the graph.
        If an edge exists with the same u and v, its attributes
        will be overwritten.
        """
        self.__graph.add_edge(edge.u, edge.v, **edge.all)

    def copy(self):
        return deepcopy(self)

    def crop(self, roi: Roi):
        """
        Will remove all nodes from self that are not contained in `roi` except for
        "dangling" nodes. This means that if there are nodes A, B s.t. there
        is an edge (A, B) and A is contained in `roi` but B is not, the edge (A, B)
        is considered contained in the `roi` and thus node B will be kept as a
        "dangling" node.

        Note there is a helper function `trim` that will remove B and replace it with
        a node at the intersection of the edge (A, B) and the bounding box of `roi`.

        Args:

            roi (:class:`Roi`):

                ROI in world units to crop to.
        """

        cropped = self.copy()

        contained_nodes = set([v.id for v in cropped.nodes if roi.contains(v.location)])
        all_contained_edges = set(
            [
                e
                for e in cropped.edges
                if e.u in contained_nodes or e.v in contained_nodes
            ]
        )
        fully_contained_edges = set(
            [
                e
                for e in all_contained_edges
                if e.u in contained_nodes and e.v in contained_nodes
            ]
        )
        partially_contained_edges = all_contained_edges - fully_contained_edges
        contained_edge_nodes = set(list(itertools.chain(*all_contained_edges)))
        all_nodes = contained_edge_nodes | contained_nodes
        dangling_nodes = all_nodes - contained_nodes

        for node in list(cropped.nodes):
            if node.id not in all_nodes:
                cropped.remove_node(node)
        for edge in list(cropped.edges):
            if edge not in all_contained_edges:
                cropped.remove_edge(edge)

        cropped.spec.roi = roi
        return cropped

    def shift(self, offset):
        for node in self.nodes:
            node.location += offset

    def new_graph(self):
        if self.directed():
            return nx.DiGraph()
        else:
            return nx.Graph()

    def trim(self, roi: Roi):
        """
        Create a copy of self and replace "dangling" nodes with contained nodes.

        A "dangling" node is defined by: Let A, B be nodes s.t. there exists an
        edge (A, B) and A is contained in `roi` but B is not. Edge (A, B) is considered
        contained, and thus B is kept as a "dangling" node.
        """

        trimmed = self.copy()

        contained_nodes = set([v.id for v in trimmed.nodes if roi.contains(v.location)])
        all_contained_edges = set(
            [
                e
                for e in trimmed.edges
                if e.u in contained_nodes or e.v in contained_nodes
            ]
        )
        fully_contained_edges = set(
            [
                e
                for e in all_contained_edges
                if e.u in contained_nodes and e.v in contained_nodes
            ]
        )
        partially_contained_edges = all_contained_edges - fully_contained_edges
        contained_edge_nodes = set(list(itertools.chain(*all_contained_edges)))
        all_nodes = contained_edge_nodes | contained_nodes
        dangling_nodes = all_nodes - contained_nodes

        next_node = 0 if len(all_nodes) == 0 else max(all_nodes) + 1

        trimmed._handle_boundaries(
            partially_contained_edges,
            contained_nodes,
            roi,
            node_id=itertools.count(next_node),
        )

        for node in trimmed.nodes:
            assert roi.contains(
                node.location
            ), f"Failed to properly contain node {node.id} at {node.location}"

        return trimmed

    def _handle_boundaries(
        self,
        crossing_edges: Iterator[Edge],
        contained_nodes: Set[int],
        roi: Roi,
        node_id: Iterator[int],
    ):
        nodes_to_remove = set([])
        for e in crossing_edges:
            u, v = self.node(e.u), self.node(e.v)
            u_in = u.id in contained_nodes
            v_in, v_out = (u, v) if u_in else (v, u)
            in_location, out_location = (v_in.location, v_out.location)
            new_location = self._roi_intercept(in_location, out_location, roi)
            if not all(np.isclose(new_location, in_location)):
                # use deepcopy because modifying this node should not modify original
                new_attrs = deepcopy(v_out.attrs)
                new_attrs["id"] = next(node_id)
                new_attrs["location"] = new_location
                new_attrs["temporary"] = True
                new_v = Node.from_attrs(new_attrs)
                new_e = Edge(
                    u=v_in.id if u_in else new_v.id, v=new_v.id if u_in else v_in.id
                )
                self.add_node(new_v)
                self.add_edge(new_e)
            nodes_to_remove.add(v_out)
        for node in nodes_to_remove:
            self.remove_node(node)

    def _roi_intercept(
        self, inside: np.ndarray, outside: np.ndarray, bb: Roi
    ) -> np.ndarray:
        """
        Given two points, one inside a bounding box and one outside,
        get the intercept between the line and the bounding box.
        """

        offset = outside - inside
        distance = np.linalg.norm(offset)
        assert not np.isclose(distance, 0), f"Inside and Outside are the same location"
        direction = offset / distance

        # `offset` can be 0 on some but not all axes leaving a 0 in the denominator.
        # `inside` can be on the bounding box, leaving a 0 in the numerator.
        # `x/0` throws a division warning, `0/0` throws an invalid warning (both are fine here)
        with np.errstate(divide="ignore", invalid="ignore"):
            bb_x = np.asarray(
                [
                    (np.asarray(bb.get_begin()) - inside) / offset,
                    (np.asarray(bb.get_end()) - inside) / offset,
                ],
                dtype=self.spec.dtype,
            )

        with np.errstate(invalid="ignore"):
            s = np.min(bb_x[np.logical_and((bb_x >= 0), (bb_x <= 1))])

        new_location = inside + s * distance * direction
        upper = np.array(bb.get_end(), dtype=self.spec.dtype)
        new_location = np.clip(
            new_location, bb.get_begin(), upper - upper * np.finfo(self.spec.dtype).eps
        )
        return new_location

    def merge(self, other, copy_from_self=False, copy=False):
        """
        Merge this graph with another. The resulting graph will have the Roi
        of the larger one.

        This only works if one of the two graphs contains the other.
        In this case, ``other`` will overwrite edges and nodes with the same
        ID in ``self`` (unless ``copy_from_self`` is set to ``True``).
        Vertices and edges in ``self`` that are contained in the Roi of ``other``
        will be removed (vice versa for ``copy_from_self``)

        A copy will only be made if necessary or ``copy`` is set to ``True``.
        """

        # It is unclear how to merge points in all cases. Consider a 10x10 graph,
        # you crop out a 5x5 area, do a shift augment, and attempt to merge.
        # What does that mean? specs have changed. It should be a new key.
        raise NotImplementedError("Merge function should not be used!")

        self_roi = self.spec.roi
        other_roi = other.spec.roi

        assert self_roi.contains(other_roi) or other_roi.contains(
            self_roi
        ), "Can not merge graphs that are not contained in each other."

        # make sure self contains other
        if not self_roi.contains(other_roi):
            return other.merge(self, not copy_from_self, copy)

        # edges and nodes in addition are guaranteed to be in merged
        base = other if copy_from_self else self
        addition = self if copy_from_self else other

        if copy:
            merged = deepcopy(base)
        else:
            merged = base

        for node in list(merged.nodes):
            if merged.spec.roi.contains(node.location):
                merged.remove_node(node)
        for edge in list(merged.edges):
            if merged.spec.roi.contains(
                merged.node(edge.u)
            ) or merged.spec.roi.contains(merged.node(edge.v)):
                merged.remove_edge(edge)
        for node in addition.nodes:
            merged.add_node(node)
        for edge in addition.edges:
            merged.add_edge(edge)

        return merged

    def to_nx_graph(self):
        """
        returns a pure networkx graph containing data from
        this Graph.
        """
        return deepcopy(self.__graph)

    @classmethod
    def from_nx_graph(cls, graph, spec):
        """
        Create a gunpowder graph from a networkx graph
        """
        if spec.directed is None:
            spec.directed = graph.is_directed()
        g = cls([], [], spec)
        g.__graph = graph
        return g

    def relabel_connected_components(self):
        """
        create a new attribute "component" for each node
        in this Graph
        """
        for i, wcc in enumerate(self.connected_components):
            for node in wcc:
                self.__graph.nodes[node]["component"] = i

    @property
    def connected_components(self):
        if not self.directed:
            return nx.connected_components(self.__graph)
        else:
            return nx.weakly_connected_components(self.__graph)

    def in_degree(self):
        return self.__graph.in_degree()

    def successors(self, node):
        if self.directed:
            return self.__graph.successors(node.id)
        else:
            return self.__graph.neighbors(node.id)

    def predecessors(self, node):
        if self.directed:
            return self.__graph.predecessors(node.id)
        else:
            return self.__graph.neighbors(node.id)


class GraphKey(Freezable):
    """A key to identify graphs in requests, batches, and across
    nodes.

    Used as key in :class:`BatchRequest` and :class:`Batch` to retrieve specs
    or graphs.

    Args:

        identifier (``string``):

            A unique, human readable identifier for this graph key. Will be
            used in log messages and to look up graphs in requests and batches.
            Should be upper case (like ``CENTER_GRAPH``). The identifier is
            unique: Two graph keys with the same identifier will refer to the
            same graph.
    """

    def __init__(self, identifier):
        self.identifier = identifier
        self.hash = hash(identifier)
        self.freeze()
        logger.debug("Registering graph type %s", self)
        setattr(GraphKeys, self.identifier, self)

    def __eq__(self, other):
        return hasattr(other, "identifier") and self.identifier == other.identifier

    def __hash__(self):
        return self.hash

    def __repr__(self):
        return self.identifier


class GraphKeys:
    """Convenience access to all created :class:`GraphKey`s. A key generated
    with::

        centers = GraphKey('CENTER_GRAPH')

    can be retrieved as::

        GraphKeys.CENTER_GRAPH
    """

    pass
