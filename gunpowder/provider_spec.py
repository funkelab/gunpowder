from freezable import Freezable
from roi import Roi

class ProviderSpec(Freezable):
    '''A possibly partial specification of a provider.
    '''

    def __init__(self):

        self.roi = Roi()
        self.has_gt = False
        self.has_gt_mask = False

        self.freeze()
