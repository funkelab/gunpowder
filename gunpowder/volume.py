from enum import Enum

from .freezable import Freezable

class VolumeType(Enum):
    RAW = 1
    GT_LABELS = 2
    GT_AFFINITIES =3
    GT_MASK = 4
    GT_IGNORE = 5
    PRED_AFFINITIES = 6

class Volume(Freezable):

    def __init__(self, data, roi, resolution, interpolate):

        self.roi = roi
        self.resolution = resolution
        self.data = data
        self.interpolate = interpolate

        self.freeze()
