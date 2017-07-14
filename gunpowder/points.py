from .freezable import Freezable
import logging

logger = logging.getLogger(__name__)

class PointsType:
    '''Describes general properties of a points type.

    Args:

        identifier (string):
            A human readable identifier for this points type. Will be used as a 
            static attribute in :class:`PointsTypes`. Should be upper case (like 
            ``PRESYN``, ``POSTSYN``).

    '''

    def __init__(self, identifier):
        self.identifier = identifier
        self.hash = hash(identifier)

    def __eq__(self, other):
        return hasattr(other, 'identifier') and self.identifier == other.identifier

    def __hash__(self):
        return self.hash

    def __repr__(self):
        return self.identifier

class PointsTypes:
    '''An expandable collection of points types, which initially contains:

        ===================  ====================================================
        identifier           purpose
        ===================  ====================================================
        ``PRESYN``           Presynaptic locations
        ``POSTSYN``          Postsynaptic locations
        ===================  ====================================================

    New points types can be added with :func:`register_points_type`.
    '''
    pass

def register_points_type(points_type):
    '''Register a new points type.

    For example, the following call::

            register_points_type(PointsType('IDENTIFIER'))

    will create a new points type available as ``PointsTypes.IDENTIFIER``. 
    ``PointsTypes.IDENTIFIER`` can then be used in dictionaries.
    '''
    logger.debug("Registering volume type " + str(points_type))
    setattr(PointsTypes, points_type.identifier, points_type)

register_points_type(PointsType('PRESYN'))
register_points_type(PointsType('POSTSYN'))


class Points(Freezable):
    def __init__(self, data, roi, resolution):
        """ Data structure to keep information about points locations within a ROI
        :param data:        a dictionary with node_ids as keys and Point instances as values
        :param roi:         Roi() (gunpowder.nodes.roi), Region of interest defined by offset and shape
        :param resolution:  n-dim tuple, list, resolution for positions of point locations 
        """
        self.data = data
        self.roi = roi
        self.resolution = resolution

        self.freeze()


class Point(Freezable):
    def __init__(self, location):
        self.location = location

        self.freeze()


class SynPoint(Point):
    def __init__(self, location, kind, location_id, synapse_id, partner_ids, props=None):
        """
        :param kind:        'PreSyn' or 'PostSyn' 
        :param location:     ndarray, [zyx]
        :param location_id:  int, unique for every synapse location across pre and postsynaptic locations 
        :param synapse_id:   int, unqiue for every synapse(synaptic partners have the same synapse_id, but different location_ids)
        :param partner_ids:  list of ints, location ids of synaptic partners
        :param props:        dict, properties
        """
        Point.__init__(self, location=location)
        self.thaw()

        self.kind         = kind
        self.location_id  = location_id
        self.synapse_id   = synapse_id
        self.partner_ids  = partner_ids
        if props is None:
            self.props = {}
        else:
            self.props = props

        self.freeze()