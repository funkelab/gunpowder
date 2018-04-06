import copy
from .freezable import Freezable

class PointsSpec(Freezable):
    '''Contains meta-information about a set of points. This is used by
    :class:`BatchProviders<BatchProvider>` to communicate the points they
    offer, as well as by :class:`Points` to describe the data they contain.

    Attributes:

        roi (:class:`Roi`):

            The region of interested represented by this set of points.
    '''

    def __init__(self, roi=None):

        self.roi = roi

        self.freeze()

    def copy(self):
        '''Create a copy of this spec.'''
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
        r += "ROI: " + str(self.roi)
        return r
