from freezable import Freezable
from enum import Enum

class VolumeType(Enum):
    RAW = 1
    GT_LABELS = 2
    GT_AFFINITIES =3
    GT_MASK = 4
    GT_IGNORE = 5
    PRED_AFFINITIES = 6

class Volume(Freezable):

    def __init__(self, data, interpolate):

        self.data = data
        self.interpolate = interpolate

        self.freeze()
