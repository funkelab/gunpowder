from .freezable import Freezable
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Points(Freezable):
    '''A list of :class:``Point``s with a specification describing the data.

    Args:

        data (dict, int->Point): A dictionary of IDs mapping to
            :class:``Point``s.

        spec (:class:`PointsSpec`): A spec describing the data.
    '''

    def __init__(self, data, spec):
        self.data = data
        self.spec = spec
        self.freeze()

class Point(Freezable):
    def __init__(self, location):
        self.location = location
        self.freeze()

class PointsKey(Freezable):
    '''A key to identify lists of points in requests, batches, and across
    nodes.

    Used as key in :class:``BatchRequest`` and :class:``Batch`` to retrieve
    specs or lists of points.

    Args:

        identifier (string):
            A unique, human readable identifier for this points key. Will be
            used in log messages and to look up points in requests and batches.
            Should be upper case (like ``CENTER_POINTS``). The identifier is
            unique: Two points keys with the same identifier will refer to the
            same points.
    '''

    def __init__(self, identifier):
        self.identifier = identifier
        self.hash = hash(identifier)
        self.freeze()
        logger.debug("Registering points type %s", self)
        setattr(PointsKeys, self.identifier, self)

    def __eq__(self, other):
        return hasattr(other, 'identifier') and self.identifier == other.identifier

    def __hash__(self):
        return self.hash

    def __repr__(self):
        return self.identifier

class PointsKeys:
    '''Convenience access to all created :class:``PointsKey``s. A key generated
    with::

        centers = PointsKey('CENTER_POINTS')

    can be retrieved as::

        PointsKeys.CENTER_POINTS
    '''
    pass
