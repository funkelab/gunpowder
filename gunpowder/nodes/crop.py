from .batch_filter import BatchFilter
import copy
import logging

logger = logging.getLogger(__name__)

class Crop(BatchFilter):
    '''Limits provided ROI to user defined ROIs per Volume-/PointsTypes 
        
    Args:
        
        volumes (dict):     Dictionary from :class:``VolumeType`` to its new :class:``ROI``.
        points (dict):      Dictionary from :class:``PointsType`` to its new :class:``ROI``.
    '''

    def __init__(self, volumes=None, points=None):

        if volumes is None:
            self.volumes = {}
        else:
            self.volumes = volumes

        if points is None:
            self.points = {}
        else:
            self.points  = points

    def setup(self):

        for crop_specs, specs in zip([self.volumes, self.points],[self.spec.volume_specs, self.spec.points_specs]):
            for type, roi in crop_specs.items():
                assert type in specs, "Asked to crop {} which is not provided".format(type)
                assert specs[type].roi.contains(roi), "Asked to Crop {} out at {} which is" \
                                            " not within provided ROI {}".format(type, roi, specs[type].roi)

        for crop_specs, specs in zip([self.volumes, self.points], [self.spec.volume_specs, self.spec.points_specs]):
            for type, roi in crop_specs.items():
                spec = specs[type].copy()
                spec.roi = roi
                self.updates(type, spec)

    def process(self, batch, request):
        pass














