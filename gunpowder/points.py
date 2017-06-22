from enum import Enum

from .freezable import Freezable

class PointsType(Enum):
    PRESYN  = 1
    POSTSYN = 2



class PointsOfType(Freezable):
    def __init__(self, data, roi, resolution):
        """ Data structure to keep information about points locations within a ROI
        :param data:        a dictionary with node_ids as keys and SynPoint instances as values
        :param roi:         Roi() (gunpowder.nodes.roi), Region of interest defined by offset and shape
        :param resolution:  n-dim tuple, list, resolution for positions of point locations 
        """
        self.data = data
        self.roi = roi
        self.resolution = resolution

        self.freeze()


class BasePoint(Freezable):
    def __init__(self, location):
        self.location = location

        self.freeze()


class SynPoint(BasePoint):
    def __init__(self, location, kind, location_id, synapse_id, partner_ids, props={}):
        """
        :param kind:        'PreSyn' or 'PostSyn' 
        :param location:     ndarray, [zyx]
        :param location_id:  int, unique for every synapse location across pre and postsynaptic locations 
        :param synapse_id:   int, unqiue for every synapse(synaptic partners have the same synapse_id, but different location_ids)
        :param partner_ids:  list of ints, location ids of synaptic partners
        :param props:        dict, properties
        """
        BasePoint.__init__(self, location=location)
        self.thaw()

        self.kind         = kind
        self.location_id  = location_id
        self.synapse_id   = synapse_id
        self.partner_ids  = partner_ids
        self.props        = props

        self.freeze()

    def get_copy(self):
        return SynPoint(kind=self.kind,
                                location=self.location,
                                location_id=self.location_id,
                                synapse_id=self.synapse_id,
                                partner_ids=self.partner_ids,
                                props=self.props)



