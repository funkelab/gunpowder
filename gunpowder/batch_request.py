from .freezable import Freezable
from .roi import Roi

class BatchRequest(Freezable):

    def __init__(self, initial_volumes=None):

        if initial_volumes is None:
            self.volumes = {}
        else:
            self.volumes = initial_volumes

        self.freeze()

        self.__center_rois()

    def add_volume_request(self, volume_type, shape):

        self.volumes[volume_type] = Roi((0,)*len(shape), shape)

        self.__center_rois()

    def get_total_roi(self):
        '''Get the union of all the requested volume ROIs.'''

        total_roi = None
        for (volume_type, roi) in self.volumes.items():
            if total_roi is None:
                total_roi = roi
            else:
                total_roi = total_roi.union(roi)
        return total_roi

    def __center_rois(self):
        '''Ensure that all ROIs are centered around the same location.'''

        total_roi = self.get_total_roi()
        if total_roi is None:
            return

        center = total_roi.get_center()

        for volume_type in self.volumes:

            roi = self.volumes[volume_type]
            self.volumes[volume_type] = roi.shift(center - roi.get_center())

    def __repr__(self):

        r = ""
        for (volume_type, roi) in self.volumes.items():
            r += "%s: %s\n"%(volume_type, roi)
        return r
