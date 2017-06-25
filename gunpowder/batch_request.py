from .freezable import Freezable
from .roi import Roi

class BatchRequest(Freezable):

    def __init__(self, initial_volumes=None, initial_points=None):

        if initial_volumes is None:
            self.volumes = {}
        else:
            self.volumes = initial_volumes

        if initial_points is None:
            self.points = {}
        else:
            self.points = initial_points

        self.freeze()

        self.__center_rois()

    def add_volume_request(self, volume_type, shape):

        self.volumes[volume_type] = Roi((0,)*len(shape), shape)

        self.__center_rois()

    def add_points_request(self, points_type, shape):

        self.points[points_type] = Roi((0,)*len(shape), shape)

        self.__center_rois()

    def get_total_roi(self):
        '''Get the union of all the requested volume ROIs.'''

        total_roi = None

        for collection_type in [self.volumes, self.points]:
            for (type, roi) in collection_type.items():
                if total_roi is None:
                    total_roi = roi
                else:
                    total_roi = total_roi.union(roi)

        return total_roi

    def get_common_roi(self):
        ''' Get the intersection of all the requested ROIs.'''

        common_roi = None

        for (volume_type, roi) in self.volumes.items():
            if common_roi is None:
                common_roi = roi
            else:
                common_roi = common_roi.intersect(roi)

        return common_roi


    def __center_rois(self):
        '''Ensure that all ROIs are centered around the same location.'''

        total_roi = self.get_total_roi()
        if total_roi is None:
            return

        center = total_roi.get_center()

        for collection_type in [self.volumes, self.points]:
            for type in collection_type:
                roi = collection_type[type]
                collection_type[type] = roi.shift(center - roi.get_center())

    def __repr__(self):

        r = ""
        for collection_type in [self.volumes, self.points]:
            for (type, roi) in collection_type.items():
                r += "%s: %s\n"%(type, roi)
        return r
