from .freezable import Freezable
from .roi import Roi

class ProviderSpec(Freezable):
    '''A possibly partial specification of a provider.
    '''

    def __init__(self):

        self.volumes = {}
        self.freeze()

    def get_total_roi(self):
        '''Get the union of all the provided volume ROIs.'''

        total_roi = None
        for (volume_type, roi) in self.volumes.items():
            if total_roi is None:
                total_roi = roi
            else:
                total_roi = total_roi.union(roi)
        return total_roi

    def __repr__(self):

        r = ""
        for (volume_type, roi) in self.volumes.items():
            r += "%s: %s\n"%(volume_type, roi)
        return r
