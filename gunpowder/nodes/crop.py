from .batch_filter import BatchFilter
import copy
import logging

logger = logging.getLogger(__name__)

class Crop(BatchFilter):
    '''Limits provided ROI to user defined ROIs per Array-/PointsKeys 
        
    Args:
        
        arrays (dict):     Dictionary from :class:``ArrayKey`` to its new :class:``ROI``.
        points (dict):      Dictionary from :class:``PointsKey`` to its new :class:``ROI``.
    '''

    def __init__(self, arrays=None, points=None):

        if arrays is None:
            self.arrays = {}
        else:
            self.arrays = arrays

        if points is None:
            self.points = {}
        else:
            self.points  = points

    def setup(self):

        for crop_specs, specs in zip([self.arrays, self.points],[self.spec.array_specs, self.spec.points_specs]):
            for key, roi in crop_specs.items():
                assert key in specs, "Asked to crop {} which is not provided".format(key)
                assert specs[key].roi.contains(roi), "Asked to Crop {} out at {} which is" \
                                            " not within provided ROI {}".format(key, roi, specs[key].roi)

        for crop_specs, specs in zip([self.arrays, self.points], [self.spec.array_specs, self.spec.points_specs]):
            for key, roi in crop_specs.items():
                spec = specs[key].copy()
                spec.roi = roi
                self.updates(key, spec)

    def process(self, batch, request):
        pass














