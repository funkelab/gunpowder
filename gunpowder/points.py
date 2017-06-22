from enum import Enum
import numpy as np

from .freezable import Freezable

class PointsType(Enum):
    PRESYN  = 1
    POSTSYN = 2



class PointsOfType(Freezable):
    def __init__(self, data, roi, resolution):
        """
        :param data:        a dictionary with node_ids as keys and SynPoint instances as values
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

    def is_inside_bb(self, bb_shape, bb_offset, margin=0):
        try:
            assert len(margin) == 3
        except:
            margin = [margin, margin, margin]

        inside_bb = True
        location  = np.asarray(self.location) - np.asarray(bb_offset)
        for dim, size in enumerate(bb_shape):
            if location[dim] < margin[dim]:
                inside_bb = False
            if location[dim] >= size - margin[dim]:
                inside_bb = False
        return inside_bb


