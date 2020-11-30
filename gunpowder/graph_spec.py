import numpy as np

import copy

from .freezable import Freezable


class GraphSpec(Freezable):
    """Contains meta-information about a graph. This is used by
    :class:`BatchProviders<BatchProvider>` to communicate the graphs they
    offer, as well as by :class:`Graph` to describe the data they contain.

    Attributes:

        roi (:class:`Roi`):

            The region of interested represented by this graph.

        directed (``bool``, optional):

            Whether the graph is directed or not.

        dtype (``dtype``, optional):

            The data type of the "location" attribute.
            Currently only supports np.float32.
    """

    def __init__(self, roi=None, directed=None, dtype=np.float32, placeholder=False):

        self.roi = roi
        self.directed = directed
        self.dtype = dtype
        self.placeholder = placeholder

        self.freeze()

    def copy(self):
        """Create a copy of this spec."""
        return copy.deepcopy(self)

    def __eq__(self, other):

        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):

        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __repr__(self):
        r = ""
        r += "ROI: " + str(self.roi) + ", "
        r += "dtype: " + str(self.dtype) + ", "
        r += "directed: " + str(self.directed) + ", "
        r += "placeholder: " + str(self.placeholder)
        return r
