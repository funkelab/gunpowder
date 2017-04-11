class ProviderSpec:
    '''A possibly partial specification of a provider.
    '''

    def __init__(self):

        self.bounding_box = None
        self.has_gt = False
        self.has_gt_mask = False

    def get_bounding_box(self):
        return self.bounding_box
