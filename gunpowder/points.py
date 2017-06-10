from enum import Enum
import numpy as np
from scipy import ndimage
import logging

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

    def get_binary_mask(self, bb_shape, bb_offset, marker='point'):
        # marker = 'point' or 'gaussian'
        binary_mask = np.zeros(bb_shape, dtype='uint8')

        for syn_point in self.data.values():
            # check for location kind
            # logging.info('offset %i %i %i' %(bb_offset[0], bb_offset[1], bb_offset[2]))
            # logging.info('shape %i %i %i' %(bb_shape[0], bb_shape[1], bb_shape[2]))
            logging.info('location %i %i %i' %(syn_point.location[0], syn_point.location[1], syn_point.location[2]))

            # if syn_point.is_inside_bb(bb_shape=bb_shape, bb_offset=bb_offset):
                # logging.info('in bounding box points.py')
                # shifted_current_loc = syn_point.location - np.asarray(bb_offset)
            shifted_current_loc = syn_point.location
            binary_mask[shifted_current_loc[0], shifted_current_loc[1], shifted_current_loc[2]] = 255
            logging.info('in matrix: %i points.py' %len(np.unique(binary_mask)))

        # return mask where location is marked as a single point
        if marker == 'point':
            return binary_mask

        # return mask where location is marked as a gaussian 'blob'
        elif marker == 'gaussian':
            binary_mask_gaussian = np.zeros_like(binary_mask, dtype='uint8')
            mask_gaussian        = ndimage.filters.gaussian_filter(binary_mask.astype(np.float32), sigma=5)
            binary_mask_gaussian[np.nonzero(mask_gaussian)] = 255
            logging.info('in matrix second: %i points.py' % len(np.unique(binary_mask_gaussian)))
            return binary_mask_gaussian



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

