from .roi import Roi
from .provider_spec import ProviderSpec
from .volume_spec import VolumeSpec
from .points_spec import PointsSpec

class BatchRequest(ProviderSpec):
    '''A collection of (possibly partial) :class:`VolumeSpec`s and 
    :class:`PointsSpec`s forming a request.

    Args:

        volume_specs (dict): A dictionary from :class:`VolumeType` to :class:`VolumeSpec`.

        points_specs (dict): A dictionary from :class:`PointsType` to :class:`PointsSpec`.
    '''

    def add_volume_request(self, volume_type, shape):
        '''Convenience method to add a volume request by providing only the 
        shape of a ROI (in world units).

        A ROI with zero-offset will be generated. If more than one request is 
        added, the ROIs with smaller shapes will be shifted to be centered in 
        the largest one.
        '''

        volume_spec = VolumeSpec()
        volume_spec.roi = Roi((0,)*len(shape), shape)

        self.volume_specs[volume_type] = volume_spec
        self.__center_rois()

    def add_points_request(self, points_type, shape):
        '''Convenience method to add a points request by providing only the 
        shape of a ROI (in world units).

        A ROI with zero-offset will be generated. If more than one request is 
        added, the ROIs with smaller shapes will be shifted to be centered in 
        the largest one.
        '''

        points_spec = PointsSpec()
        points_spec.roi = Roi((0,)*len(shape), shape)

        self.points_specs[points_type] = points_spec
        self.__center_rois()

    def __center_rois(self):
        '''Ensure that all ROIs are centered around the same location.'''

        total_roi = self.get_total_roi()
        if total_roi is None:
            return

        center = total_roi.get_center()

        for specs_type in [self.volume_specs, self.points_specs]:
            for type in specs_type:
                roi = specs_type[type].roi
                specs_type[type].roi = roi.shift(center - roi.get_center())
