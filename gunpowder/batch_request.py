from .points import PointsType
from .points_spec import PointsSpec
from .provider_spec import ProviderSpec
from .roi import Roi
from .volume import VolumeType
from .volume_spec import VolumeSpec

class BatchRequest(ProviderSpec):
    '''A collection of (possibly partial) :class:`VolumeSpec`s and
    :class:`PointsSpec`s forming a request.

    For usage, see the documentation of :class:`ProviderSpec`.
    '''

    def add(self, identifier, shape):
        '''Convenience method to add a volume or point spec by providing only
        the shape of a ROI (in world units).

        A ROI with zero-offset will be generated. If more than one request is
        added, the ROIs with smaller shapes will be shifted to be centered in
        the largest one.
        '''

        if isinstance(identifier, VolumeType):
            spec = VolumeSpec()
        elif isinstance(identifier, PointsType):
            spec = PointsSpec()
        else:
            raise RuntimeError("Only VolumeType or PointsType can be added.")

        spec.roi = Roi((0,)*len(shape), shape)

        self[identifier] = spec
        self.__center_rois()

    def __center_rois(self):
        '''Ensure that all ROIs are centered around the same location.'''

        total_roi = self.get_total_roi()
        if total_roi is None:
            return

        center = total_roi.get_center()

        for specs_type in [self.volume_specs, self.points_specs]:
            for identifier in specs_type:
                roi = specs_type[identifier].roi
                specs_type[identifier].roi = roi.shift(center - roi.get_center())
