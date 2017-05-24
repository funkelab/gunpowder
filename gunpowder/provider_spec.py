from .freezable import Freezable
from .roi import Roi

class ProviderSpec(Freezable):
    '''A possibly partial specification of a provider.
    '''

    def __init__(self):

        self.roi = Roi()
        self.gt_roi = None
        self.has_gt = False
        self.has_gt_mask = False

        self.freeze()

    def __repr__(self):

        r  = "raw ROI    : " + str(self.roi) + "\n"
        r += "GT  ROI    : " + str(self.gt_roi) + "\n"
        r += "has GT     : " + str(self.has_gt) + "\n"
        r += "has GT mask: " + str(self.has_gt_mask)

        return r
