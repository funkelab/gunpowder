from enum import Enum

from .freezable import Freezable

class VolumeType(Enum):
    RAW = 1
    ALPHA_MASK = 2
    GT_LABELS = 3
    GT_AFFINITIES = 4
    GT_MASK = 5
    GT_IGNORE = 6
    PRED_AFFINITIES = 7
    LOSS_GRADIENT = 8
    GT_BM_PRESYN = 9
    GT_BM_POSTSYN = 10


class Volume(Freezable):

    def __init__(self, data, roi, resolution, interpolate):

        self.roi = roi
        self.resolution = resolution
        self.data = data
        self.interpolate = interpolate

        self.freeze()
