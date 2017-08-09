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

        self.upstream_spec = self.get_upstream_provider().get_spec()
        self.spec = copy.deepcopy(self.upstream_spec)

        for crop_specs, specs in zip([self.volumes, self.points],[self.spec.volumes, self.spec.points]):
            for type, roi in crop_specs.items():
                assert type in specs, "Asked to crop {} which is not provided".format(type)
                assert specs[type].contains(roi), "Asked to Crop {} out at {} which is" \
                                            " not within provided ROI {}".format(type, roi, specs[type])

        for crop_specs, specs in zip([self.volumes, self.points], [self.spec.volumes, self.spec.points]):
            for type, roi in crop_specs.items():
                specs[type] = roi

    def get_spec(self):
        return self.spec

    def prepare(self, request):
        pass

    def process(self, batch, request):
        pass














