from .graph_spec import GraphSpec
from .roi import Roi
from .coordinate import Coordinate
from .freezable import Freezable

import numpy as np
import networkx as nx

from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Set, Iterator, Any
import logging
import itertools


logger = logging.getLogger(__name__)


class Vertex(Freezable):
    """
    A stucture representing each vertex in a Graph.

    Args:

        id (``int``):

            A unique identifier for this Vertex

        location (``np.ndarray``):

            A numpy array containing a nodes location

        Optional attrs (``dict``, str -> ``Any``):

            A dictionary containing a mapping from attribute to value.
            Used to store any extra attributes associated with the
            Vertex such as color, size, etc.

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
        self.__id = id
        self.__location = location
        self.__temporary = temporary
        self.attrs = attrs
        self.freeze()

    @property
    def location(self):
        return self.__location

    @location.setter
    def location(self, new_location):
        self.__location = new_location

    @property
    def attrs(self):
        return self.__attrs

    @attrs.setter
    def attrs(self, attrs):
        self.__attrs = attrs if attrs is not None else {}

    @property
    def id(self):
        return self.__id

    @property
    def original_id(self):
        return self.id if not self.temporary else None

    @property
    def temporary(self):
        return self.__temporary

    @property
    def all(self):
        data = self.__attrs
        data["id"] = self.id
        data["location"] = self.location
        data["temporary"] = self.temporary
        return data

    @classmethod
    def from_attrs(cls, attrs: Dict[str, Any]):
        special_attrs = ["id", "location", "temporary"]
        vertex_id = attrs["id"]
        location = attrs["location"]
        temporary = attrs["temporary"]
        remaining_attrs = {k: v for k, v in attrs.items() if k not in special_attrs}
        return cls(
            id=vertex_id, location=location, temporary=temporary, attrs=remaining_attrs
        )

    def __str__(self):
        return f"Vertex({self.temporary}) ({self.id}) at ({self.location})"

    def __repr__(self):
        return str(self)


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
    """A structure containing a list of :class:`Vertex`, a list of :class:'Edge',
    and a specification describing the data.

    Args:

        vertices (``iterator``, :class:`Vertex`):

            An iterator containing Vertices.

        edges (``iterator``, :class:`Edge`):

            An iterator containing Edges.

        spec (:class:`GraphSpec`):

            A spec describing the data.
    """

    def __init__(
        self, vertices: Iterator[Vertex], edges: Iterator[Edge], spec: GraphSpec
    ):
        self.__spec = spec
        self.__graph = self.create_graph(vertices, edges)

    @property
    def spec(self):
        return self.__spec

    @property
    def directed(self):
        return self.spec.directed

    def create_graph(self, vertices: Iterator[Vertex], edges: Iterator[Edge]):
        if self.directed:
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()

        vs = [(v.id, v.all) for v in vertices]
        graph.add_nodes_from(vs)
        graph.add_edges_from([(e.u, e.v, e.all) for e in edges])
        return graph

    @property
    def vertices(self):
        for vertex_id, vertex_attrs in self.__graph.nodes.items():
            yield Vertex.from_attrs(vertex_attrs)

    def num_vertices(self):
        return self.__graph.number_of_nodes()

    @property
    def edges(self):
        for (u, v), attrs in self.__graph.edges.items():
            yield Edge(u, v, attrs)

    def __str__(self):
        string = "Vertices:\n"
        for vertex in self.vertices:
            string += f"{vertex}\n"
        string += "Edges:\n"
        for edge in self.edges:
            string += f"{edge}\n"
        return string

    def __repr__(self):
        return str(self)

    def vertex(self, id: int):
        attrs = self.__graph.nodes[id]
        return Vertex.from_attrs(attrs)

    def remove_vertex(self, vertex: Vertex):
        self.__graph.remove_node(vertex.id)

    def add_vertex(self, vertex: Vertex):
        """
        Adds a vertex to the graph.
        If a vertex exists with the same id as the vertex you are adding,
        its attributes will be overwritten.
        """
        self.__graph.add_node(vertex.id, **vertex.all)

    def remove_edge(self, edge: Edge):
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

    def crop(self, roi: Roi, copy: bool = False):
        """
        Will remove all vertices from self that are not contained in `roi` except for
        "dangling" vertices. This means that if there are vertices A, B s.t. there
        is an edge (A, B) and A is contained in `roi` but B is not, the edge (A, B)
        is considered contained in the `roi` and thus vertex B will be kept as a
        "dangling" vertex.

        Note there is a helper function `trim` that will remove B and replace it with
        a node at the intersection of the edge (A, B) and the bounding box of `roi`.
        """

        # Current implementation removes nodes from cropped if outside the roi,
        # Thus if copy is set to False, it actually modifies the input structre,
        # rather than just providing a view into a subset of the data
        copy = True

        if copy:
            cropped = self.copy()
        else:
            cropped = self
        cropped.__spec = self.__spec

        contained_nodes = set(
            [v.id for v in cropped.vertices if roi.contains(v.location)]
        )
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

        for vertex in list(cropped.vertices):
            if vertex.id not in all_nodes:
                cropped.remove_vertex(vertex)
        for edge in list(cropped.edges):
            if edge not in all_contained_edges:
                cropped.remove_edge(edge)

        return cropped

    def shift(self, offset):
        for vertex in self.vertices:
            vertex.location += offset

    def new_graph(self):
        if self.directed():
            return nx.DiGraph()
        else:
            return nx.Graph()

    def trim(self, roi: Roi):
        """
        Create a copy of self and replace "dangling" vertices with contained vertices.

        A "dangling" vertex is defined by: Let A, B be vertices s.t. there exists an
        edge (A, B) and A is contained in `roi` but B is not. Edge (A, B) is considered
        contained, and thus B is kept as a "dangling" vertex.
        """

        trimmed = self.copy()

        contained_nodes = set(
            [v.id for v in trimmed.vertices if roi.contains(v.location)]
        )
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

        trimmed._handle_boundaries(
            partially_contained_edges,
            contained_nodes,
            roi,
            node_id=itertools.count(max(all_nodes) + 1),
        )

        return trimmed

    def _handle_boundaries(
        self,
        crossing_edges: Iterator[Edge],
        contained_nodes: Set[int],
        roi: Roi,
        node_id: Iterator[int],
    ):
        new_points = []
        new_edges = []
        for e in crossing_edges:
            u, v = self.vertex(e.u), self.vertex(e.v)
            u_in = u.id in contained_nodes
            v_in, v_out = (u, v) if u_in else (v, u)
            in_location, out_location = (v_in.location, v_out.location)
            new_location = self._roi_intercept(in_location, out_location, roi)
            if not all(np.isclose(new_location, in_location)):
                # use deepcopy because modifying this vertex should not modify original
                new_attrs = deepcopy(v_out.attrs)
                new_v = Vertex(
                    id=next(node_id),
                    location=new_location,
                    attrs=new_attrs,
                    temporary=True,
                )
                new_e = Edge(
                    u=v_in.id if u_in else new_v.id, v=new_v.id if u_in else v_in.id
                )
                self.add_vertex(new_v)
                self.add_edge(new_e)
            self.remove_edge(e)
            self.remove_vertex(v_out)
        return new_points, new_edges

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
                dtype=float,
            )

        with np.errstate(invalid="ignore"):
            s = np.min(bb_x[np.logical_and((bb_x >= 0), (bb_x <= 1))])

        new_location = inside + s * distance * direction
        new_location = np.clip(new_location, bb.get_begin(), bb.get_end())
        if not bb.contains_point(new_location):
            raise Exception(
                (
                    "Roi {} does not contain point {}!\n"
                    + "inside {}, outside: {}, distance: {}, direction: {}, s: {}"
                ).format(bb, new_location, inside, outside, distance, direction, s)
            )
        return new_location

    def merge(self, other, copy_from_self=False, copy=False):
        """
        Merge this graph with another. The resulting graph will have the Roi
        of the larger one.

        This only works if one of the two graphs contains the other.
        In this case, ``other`` will overwrite edges and vertices with the same
        ID in ``self`` (unless ``copy_from_self`` is set to ``True``).
        Vertices and edges in ``self`` that are contained in the Roi of ``other``
        will be removed (vice versa for ``copy_from_self``)

        A copy will only be made if necessary or ``copy`` is set to ``True``.
        """

        self_roi = self.spec.roi
        other_roi = other.spec.roi

        assert self_roi.contains(other_roi) or other_roi.contains(
            self_roi
        ), "Can not merge point sets that are not contained in each other."

        # make sure self contains other
        if not self_roi.contains(other_roi):
            return other.merge(self, not copy_from_self, copy)

        # edges and vertices in addition are guaranteed to be in merged
        base = other if copy_from_self else self
        addition = self if copy_from_self else other

        if copy:
            merged = deepcopy(base)
        else:
            merged = base

        for vertex in list(merged.vertices):
            if merged.spec.roi.contains_point(vertex.location):
                merged.remove_vertex(vertex)
        for edge in list(merged.edges):
            if merged.spec.roi.contains_point(
                merged.vertex(edge.u)
            ) or merged.spec.roi.contains_point(merged.vertex(edge.v)):
                merged.remove_edge(edge)
        for vertex in addition.vertices:
            merged.add_vertex(vertex)
        for edge in addition.edges:
            merged.add_edge(edge)

        return merged


class GraphKey(Freezable):
    """A key to identify graphs in requests, batches, and across
    nodes.

    Used as key in :class:`BatchRequest` and :class:`Batch` to retrieve specs
    or graphs.

    Args:

        identifier (``string``):

            A unique, human readable identifier for this graph key. Will be
            used in log messages and to look up points in requests and batches.
            Should be upper case (like ``CENTER_GRAPH``). The identifier is
            unique: Two graph keys with the same identifier will refer to the
            same graphs.
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

        centers = GraphKey('CENTER_POINTS')

    can be retrieved as::

        GraphKeys.CENTER_POINTS
    """

    pass


'''
from .points_base import PointsBase, PointBase, PointsKey, PointsKeys
from .points_spec import PointsSpec
from .roi import Roi
from .coordinate import Coordinate

from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Set
import logging
import numpy as np
import networkx as nx


logger = logging.getLogger(__name__)


class SpatialGraph(nx.DiGraph):
    """
    An extension of a DiGraph that assumes each point has a spatial coordinate.
    Adds utility functions such as cropping to an ROI, shifting all points by
    an offset, and relabelling connected components.
    """

    def crop(self, roi: Roi, copy: bool = False, relabel_nodes=False):
        """
        Remove all nodes not in this roi.
        """

        # Copy self if needed
        if copy:
            cropped = deepcopy(self)
        else:
            cropped = self

        if len(cropped.nodes) == 0:
            return cropped

        # Group nodes based on location
        all_nodes = set(cropped.nodes.keys())
        to_keep = set(
            key
            for key in cropped.nodes.keys()
            if roi.contains(cropped.nodes[key]["location"])
        )
        to_remove = all_nodes - to_keep

        # Get new boundary nodes and edges
        new_nodes, new_edges = self._handle_boundaries(
            to_keep, roi, next_node_id=max(all_nodes) + 1
        )

        # Handle node and edge changes
        for node in to_remove:
            cropped.remove_node(node)
        for node_id, attrs in new_nodes.items():
            cropped.add_node(node_id, **attrs)
        for u, v in new_edges:
            if u not in cropped.nodes or v not in cropped.nodes:
                raise Exception("Trying to add an edge between non-existant points!")
            cropped.add_edge(u, v)

        if relabel_nodes:
            cropped = nx.convert_node_labels_to_integers(cropped)

        return cropped

    def crop_out(self, roi: Roi, copy: bool = False, relabel_nodes=False):
        """
        Remove all nodes in this roi.
        """

        # Copy self if needed
        if copy:
            cropped = deepcopy(self)
        else:
            cropped = self

        if len(cropped.nodes) == 0:
            return cropped

        # Group nodes based on location
        all_nodes = set(cropped.nodes.keys())
        to_remove = set(
            key
            for key in cropped.nodes.keys()
            if roi.contains(cropped.nodes[key]["location"])
        )
        to_keep = all_nodes - to_remove

        # Get new boundary nodes and edges
        new_nodes, new_edges = self._handle_boundaries(
            to_keep, roi, next_node_id=max(all_nodes) + 1
        )

        # Handle node and edge changes
        for node in to_remove:
            cropped.remove_node(node)
        for node_id, attrs in new_nodes.items():
            cropped.add_node(node_id, **attrs)
        for u, v in new_edges:
            if u not in cropped.nodes or v not in cropped.nodes:
                raise Exception("Trying to add an edge between non-existant points!")
            cropped.add_edge(u, v)

        if relabel_nodes:
            cropped = nx.convert_node_labels_to_integers(cropped)

        return cropped

    def merge(self, other, combine_overlapping=True):
        """
        Merge this graph with another graph.
        Each PointBase will recieve a new id in range
        0, ..., len(self.nodes) + len(other.nodes)

        if combine_overlapping = True, nodes that
        share the same location will be merged, keeping
        edges from both.
        """
        combined = nx.disjoint_union(self, other)

        if combine_overlapping:
            combined.merge_overlapping_points()

        return combined

    

    def merge_overlapping_points(self):
        # TODO: this could probably be improved by using scipy.spatial.cKDTree.

        """
        it is not enough to just check location of a single node for overlap,
        multiple neurons that get close to each other might get merged.
        Instead, check that the neighbors overlap as well. for smaller crops,
        some neighbors may be cut off. So check that for any two points,
        they are the same if their locations match, but also that the union
        of the sets of neighbor locations for both nodes, is no longer than
        the length of the larger set of neighbor locations. i.e. they contain
        only neighbors that overlap, or are unseen by the potential match.
        """
        locations = {}
        replacements = {}
        for node_id, node_attrs in self.nodes.items():
            loc = node_attrs["location"]
            # convert to int to get hashable value
            # multiply by 1000 to get higher precision than rounding
            loc = tuple(int(x * 1000) for x in loc)
            if loc not in locations:
                locations[loc] = {node_id: set()}
                for pred in self.pred[node_id]:
                    pred_loc = self.nodes[pred]["location"]
                    pred_loc = tuple(int(x * 1000) for x in pred_loc)
                    locations[loc][node_id].add(pred_loc)
                for succ in self.succ[node_id]:
                    succ_loc = self.nodes[succ]["location"]
                    succ_loc = tuple(int(x * 1000) for x in succ_loc)
                    locations[loc][node_id].add(succ_loc)
            else:
                neighbor_locations = set()
                for pred in self.pred[node_id]:
                    pred_loc = self.nodes[pred]["location"]
                    pred_loc = tuple(int(x * 1000) for x in pred_loc)
                    neighbor_locations.add(pred_loc)
                for succ in self.succ[node_id]:
                    succ_loc = self.nodes[succ]["location"]
                    succ_loc = tuple(int(x * 1000) for x in succ_loc)
                    neighbor_locations.add(succ_loc)
                matched = False
                for potential_match, neighbor_locs in locations[loc].items():
                    if len(neighbor_locs | neighbor_locations) <= max(
                        len(neighbor_locs), len(neighbor_locations)
                    ):
                        replacements[node_id] = potential_match
                        locations[loc][potential_match] = (
                            neighbor_locs | neighbor_locations
                        )
                        matched = True
                if not matched:
                    locations[loc][node_id] = neighbor_locations
        nx.relabel_nodes(self, replacements, copy=False)
        self.remove_edges_from(nx.selfloop_edges(self))

    def shift(self, offset: Coordinate):
        """
        Shift every node's location by some offset
        """
        for point_attrs in self.nodes.values():
            point_attrs["location"] += offset

    def _relabel_connected_components(self):
        """
        Assign a label to each connected component
        """
        for i, connected_component in enumerate(nx.weakly_connected_components(self)):
            for point in connected_component:
                self.nodes[point]["component"] = i
        return self

    def _handle_boundaries(self, contained: Set[int], roi: Roi, next_node_id: int):
        new_points = {}
        new_edges = []
        for u, v in self.edges:
            u_in, v_in = u in contained, v in contained
            if u_in != v_in:
                in_id, out_id = (u, v) if u_in else (v, u)
                in_attrs, out_attrs = (self.nodes[in_id], self.nodes[out_id])
                new_location = self._roi_intercept(
                    in_attrs["location"], out_attrs["location"], roi
                )
                if not all(np.isclose(new_location, in_attrs["location"])):
                    new_attrs = deepcopy(out_attrs)
                    new_attrs["location"] = new_location
                    new_points[next_node_id] = new_attrs
                    new_edges.append(
                        (in_id, next_node_id) if u_in else (next_node_id, in_id)
                    )
                    next_node_id += 1
        return new_points, new_edges

    def _roi_intercept(
        self, inside: np.ndarray, outside: np.ndarray, bb: Roi
    ) -> np.ndarray:
        """
        Given two points, one inside a bounding box and one outside,
        get the intercept between the line and the bounding box.
        """

        offset = outside - inside
        distance = np.linalg.norm(offset)
        assert not np.isclose(distance, 0), "Offset cannot be zero"
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
                dtype=float,
            )

        with np.errstate(invalid="ignore"):
            s = np.min(bb_x[np.logical_and((bb_x >= 0), (bb_x <= 1))])

        # TODO: Fix this, it seems unsound
        # subtract a small amount from distance to round towards "inside" rather
        # than attempting to round down if too high and up if too low.
        new_location = np.floor(inside + s * distance * direction)
        new_location = np.clip(
            new_location, bb.get_begin(), bb.get_end() - Coordinate([1, 1, 1])
        )
        if not bb.contains(new_location):
            raise Exception(
                (
                    "Roi {} does not contain point {}!\n"
                    + "inside {}, outside: {}, distance: {}, direction: {}, s: {}"
                ).format(bb, new_location, inside, outside, distance, direction, s)
            )
        return new_location


class Point(PointBase):
    """An extension of ``PointBase`` that allows arbitrary attributes
    to be defined on each point in a networkx friendly way.

    Args:

        location (array-like of ``float``):

            The location of this point.
    """

    def __init__(self, location, **kwargs):
        super().__init__(location)
        self.thaw()
        self.kwargs = kwargs
        self.freeze()

    @property
    def attrs(self):
        attrs = {}
        attrs.update(self.kwargs)
        attrs["location"] = self.location
        return attrs

    def __repr__(self):
        return str(self.location)

    def copy(self):
        return Point(self.location, **self.kwargs)


class Points(PointsBase):
    """A subclass of points that supports edges between points.
    Uses a networkx DiGraph to store points and cropping the graph
    generates nodes at the intersection of the boundary box and edges
    in the graph.

    Differences between PointsGraph and PointsBase:
    - crop():
        will create new nodes along edges crossing the roi.
        Given adjacent nodes `a`,`b` with `a` in the `roi`, and `b` out of the `roi`,
        a new node `c` will be created intersecting the `roi` between
        `a` and `b`. `c` will not use the same id as `b` to avoid problems
        encountered when two `inside` nodes are adjacent to one `outside` node

    Args:

        data (``dict``, ``int`` -> :class:`PointBase`):

            A dictionary of IDs mapping to :class:`PointsBase<PointBase>`.

        edges (``list``, ``tuple``, ``int``):

            A list of ID pairs (a,b) denoting edges from a to b.

        spec (:class:`PointsSpec`):

            A spec describing the data.
    """

    def __init__(
        self,
        data: Dict[int, PointBase],
        spec: PointsSpec,
        edges: Optional[List[Tuple[int, int]]] = None,
    ):
        self.spec = spec
        self._graph = self._initialize_graph(data, edges)
        self.freeze()

    @property
    def graph(self) -> SpatialGraph:
        return self._graph

    @property
    def data(self) -> Dict[int, PointBase]:
        return self._graph_to_points()

    def _initialize_graph(
        self,
        data: Optional[Dict[int, PointBase]] = None,
        edges: Optional[List[Tuple[int, int]]] = None,
    ):
        self._graph = SpatialGraph()
        if data is not None:
            self._add_points(data)
        if edges is not None:
            self._add_edges(edges)
        return self._graph

    def crop(self, roi: Roi, copy: bool = False):
        """
        Crop this point set to the given ROI.
        """

        if copy:
            cropped = deepcopy(self)
        else:
            cropped = self

        # Crop the graph representation of the point set
        cropped._graph.crop(roi)

        # Override the current roi with the original crop roi
        cropped.spec.roi = roi
        return cropped

    def loose_merge(self, points, copy_from_self=False, copy=False):
        """
        Merge two sets of points together roi's which may or may not overlap
        partially or fully.
        The Roi of self will grow to contain all points, and any points in self
        that have the same id as points in points will be overwritten.
        """
        if self is points or self._graph is points._graph:
            return self

        self.spec.roi = self.spec.roi.union(points.spec.roi)
        merged_graph = self._graph.merge(points._graph)

        # replace points
        if copy:
            merged = deepcopy(self)
            merged._graph = deepcopy(merged_graph)
        else:
            merged = self
            merged._graph = merged_graph

        return merged

    def merge(self, points, copy_from_self=False, copy=False):
        """Merge these points with another set of points. The resulting points
        will have the ROI of the larger one.

        This only works if one of the two point sets is contained in the other.
        In this case, ``points`` will overwrite points with the same ID in
        ``self`` (unless ``copy_from_self`` is set to ``True``).

        A copy will only be made if necessary or ``copy`` is set to ``True``.
        """
        if self is points or self._graph is points._graph:
            return self

        self_roi = self.spec.roi
        points_roi = points.spec.roi

        assert self_roi.contains(points_roi) or points_roi.contains(
            self_roi
        ), "Can not merge point sets that are not contained in each other."

        # make sure self contains points
        if not self_roi.contains(points_roi):
            return points.merge(self, not copy_from_self, copy)

        # crop out points in roi from self, replace them with new points
        self._graph.crop_out(points_roi)
        merged_graph = self._graph.merge(points._graph)

        # replace points
        if copy:
            merged = deepcopy(self)
            merged._graph = deepcopy(merged_graph)
        else:
            merged = self
            merged._graph = merged_graph

        return merged

    def remove(self, point_id: int):
        """
        Removes a node with id `point_id` if it exists.
        for every pair of different points a, b that exists s.t.
        there exists edges (a, point_id) and (point_id, b)
        will result in an edge (a, b) being added.
        """
        if point_id in self._graph.nodes:
            preds = self._graph.pred[point_id]
            succs = self._graph.succ[point_id]
            for pred in preds:
                for succ in succs:
                    if pred != succ:
                        self._graph.add_edge(pred, succ)
            self._graph.remove_node(point_id)

    def disjoint_merge(self, other):
        """
        Merge two graphs from different sources. i.e. none of the nodes
        should overwrite each other
        """
        g1, g2 = self._graph, other._graph
        g = g1.merge(g2)
        return Points._from_graph(g)

    def _add_points(self, data: Dict[int, PointBase]):
        """
        Add points from the old data format: Dict from node id to PointBase,
        into the new graph.
        """
        for point_id, point in data.items():
            loc = point.location
            if isinstance(point, Point) and point.kwargs:
                self._graph.add_node(
                    point_id, location=deepcopy(loc), **deepcopy(point.kwargs)
                )
            else:
                self._graph.add_node(point_id, location=deepcopy(loc))

    def _add_edges(self, edges: List[Tuple[int, int]]):
        """
        Add list of edges in id pair tuples into the graph.
        """
        for u, v in edges:
            if u not in self._graph.nodes or v not in self._graph.nodes:
                logging.warning(
                    (
                        "{} is{} in the graph, {} is{} in the graph, "
                        + "thus an edge cannot be added between them"
                    ).format(
                        u,
                        "" if u in self._graph.nodes else " not",
                        v,
                        "" if v in self._graph.nodes else " not",
                    )
                )
                raise Exception(
                    "This graph does not contain a point with id {}! The edge {} is invalid".format(
                        v if u in self._graph.nodes else u
                    )
                )
            else:
                self._graph.add_edge(u, v)

    def _graph_to_points(self) -> Dict[int, Point]:
        """
        Turn a graph into the old data format of Dict mapping from
        node_id to PointBase class
        """
        point_data = {}
        for point_id, point_attrs in self._graph.nodes.items():
            # do not deep copy location here. Modifying an attribute on the
            # point needs to modify that attribute on the graph
            attrs = deepcopy(point_attrs)
            loc = attrs.pop("location")
            point_data[point_id] = Point(location=loc, **attrs)
        return point_data

    def _update_graph(self, points: Dict[int, PointBase]):
        """
        Add a Dict mapping node_ids to PointsBase to the graph,
        if they are not already contained.
        """
        for point_id, point in points.items():
            if point_id not in self._graph.nodes:
                continue
            if isinstance(point, Point):
                self._graph.nodes[point_id].update(point.attrs)
            elif isinstance(point, PointBase):
                self._graph.nodes[point_id]["location"] = point.location

    @classmethod
    def _from_graph(cls, graph: nx.Graph, spec: PointsSpec):
        """
        Create a new Points datastructure from a networkx graph.
        """
        graph = graph.to_directed()
        x = cls({}, spec)
        graph.__class__ = SpatialGraph
        x._graph = graph
        return x
'''
