from .freezable import Freezable

class PointsSpec(Freezable):
    '''Contains meta-information about a set of points. This is used by 
    :class:`BatchProvider`s to communicate the points they offer, as well as by 
    :class:`Points`s to describe the data they contain.

    Attributes:

        roi (:class:`Roi`): The region of interested represented by this set of 
        points. Can be `None` for `BatchProvider`s that allow requests for 
        volumes everywhere, but will always be set for points specs that are 
        part of a :class:`Points` set.
    '''

    def __init__(self):

        self.roi = None

        self.freeze()

    def __repr__(self):
        r = ""
        r += "roi: " + str(self.roi)
        return r
