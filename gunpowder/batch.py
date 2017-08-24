import logging
import multiprocessing

from .freezable import Freezable
from .profiling import ProfilingStats

logger = logging.getLogger(__name__)

class Batch(Freezable):
    '''Contains the requested batch.
    '''

    __next_id = multiprocessing.Value('L')

    @staticmethod
    def get_next_id():
        with Batch.__next_id.get_lock():
            next_id = Batch.__next_id.value
            Batch.__next_id.value += 1
        return next_id

    def __init__(self):

        self.id = Batch.get_next_id()
        self.profiling_stats = ProfilingStats()
        self.volumes = {}
        self.points  = {}
        self.affinity_neighborhood = None
        self.loss = None
        self.iteration = None

        self.freeze()

    def get_total_roi(self):
        '''Get the union of all the volume ROIs in the batch.'''

        total_roi = None

        for collection_type in [self.volumes, self.points]:
            for (type, obj) in collection_type.items():
                if total_roi is None:
                    total_roi = obj.spec.roi
                else:
                    total_roi = total_roi.union(obj.spec.roi)

        return total_roi

    def __repr__(self):

        r = ""
        for collection_type in [self.volumes, self.points]:
            for (type, obj) in collection_type.items():
                r += "%s: %s\n"%(type, obj.spec)
        return r
