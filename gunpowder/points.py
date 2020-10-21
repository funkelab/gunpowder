from .freezable import Freezable
from .graph import Graph, Node, GraphKey, GraphKeys

import logging
import warnings
import numpy as np

logger = logging.getLogger(__name__)


class Points(Graph):
    """An Alias of Graph that supports the points API

    Args:

        data (``dict``, ``int`` -> :class:`Point`):

            A dictionary of IDs mapping to :class:`Points<Point>`.

        spec (:class:`PointsSpec`):

            A spec describing the data.
    """

    def __init__(self, data, spec):
        warnings.warn("Points are depricated. Please use Graph")
        nodes = [Node(id=i, location=p.location) for i, p in data.items()]
        super().__init__(nodes, [], spec)
        self.__spec = spec
        self.freeze()

    @property
    def data(self):
        return {v.id: Point(v.location) for v in self.nodes}

    @property
    def directed(self):
        return True

    def __repr__(self):
        return "%s, %s" % (self.data, self.spec)


class Point(Freezable):
    """A point with a location, as stored in :class:`Points`.

    Args:

        location (array-like of ``float``):

            The location of this point.
    """

    def __init__(self, location):
        self.location = np.array(location, dtype=np.float32)
        self.freeze()

    def __repr__(self):
        return str(self.location)

    def copy(self):
        return Point(self.location)


PointsKey = GraphKey

PointsKeys = GraphKeys
