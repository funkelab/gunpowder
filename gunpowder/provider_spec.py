from .freezable import Freezable
from .roi import Roi

class ProviderSpec(Freezable):
    '''A possibly partial specification of a provider.
    '''

    def __init__(self):

        self.volumes = {}
        self.points  = {}
        self.freeze()

    def get_total_roi(self):
        '''Get the union of all the provided volume ROIs.'''

        total_roi = None
        for collection_type in [self.volumes, self.points]:
            for (type, roi) in collection_type.items():
                if total_roi is None:
                    total_roi = roi
                else:
                    total_roi = total_roi.union(roi)
            return total_roi


    def __repr__(self):

        r = ""
        for collection_type in [self.volumes, self.points]:
            for (type, roi) in collection_type.items():
                r += "%s: %s\n"%(type, roi)
        return r
