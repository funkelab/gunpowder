from freezable import Freezable
from profiling import ProfilingStats

class Batch(Freezable):
    '''Contains the requested batch.
    '''

    def __init__(self, batch_spec):

        self.spec = batch_spec
        self.profiling_stats = ProfilingStats()
        self.raw = None
        self.gt = None
        self.gt_mask = None
        self.gt_affinities = None
        self.gt_offset = None
        self.affinity_neighborhood = None
        self.prediction = None
        self.gradient = None
        self.loss = None

        self.freeze()
