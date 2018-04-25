import logging
import multiprocessing

from .freezable import Freezable
from .profiling import ProfilingStats
from .array import Array, ArrayKey
from .points import Points, PointsKey

logger = logging.getLogger(__name__)

class Batch(Freezable):
    '''Contains the requested batch as a collection of :class:`Arrays<Array>`
    and :class:`Points` that is passed through the pipeline from sources to
    sinks.

    This collection mimics a dictionary. Items can be added with::

        batch = Batch()
        batch[array_key] = Array(...)
        batch[points_key] = Points(...)

    Here, ``array_key`` and ``points_key`` are :class:`ArrayKey` and
    :class:`PointsKey`. The items can be queried with::

        array = batch[array_key]
        points = batch[points_key]

    Furthermore, pairs of keys/values can be iterated over using
    ``batch.items()``.

    To access only arrays or point sets, use the dictionaries ``batch.arrays``
    or ``batch.points``, respectively.

    Attributes:

        arrays (dict from :class:`ArrayKey` to :class:`Array`):

            Contains all arrays that have been requested for this batch.

        points (dict from :class:`PointsKey` to :class:`Points`):

            Contains all point sets that have been requested for this batch.
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
        self.arrays = {}
        self.points  = {}
        self.affinity_neighborhood = None
        self.loss = None
        self.iteration = None

        self.freeze()

    def __setitem__(self, key, value):

        if isinstance(value, Array):
            assert isinstance(key, ArrayKey), (
                "Only a ArrayKey is allowed as key for an Array value.")
            self.arrays[key] = value

        elif isinstance(value, Points):
            assert isinstance(key, PointsKey), (
                "Only a PointsKey is allowed as key for a Points value.")
            self.points[key] = value

        else:
            raise RuntimeError(
                "Only Array or Points can be set in a %s."%type(self).__name__)

    def __getitem__(self, key):

        if isinstance(key, ArrayKey):
            return self.arrays[key]

        elif isinstance(key, PointsKey):
            return self.points[key]

        else:
            raise RuntimeError(
                "Only ArrayKey or PointsKey can be used as keys in a "
                "%s."%type(self).__name__)

    def __len__(self):

        return len(self.arrays) + len(self.points)

    def __contains__(self, key):

        if isinstance(key, ArrayKey):
            return key in self.arrays

        elif isinstance(key, PointsKey):
            return key in self.pointss

        else:
            raise RuntimeError(
                "Only ArrayKey or PointsKey can be used as keys in a "
                "%s."%type(self).__name__)

    def __delitem__(self, key):

        if isinstance(key, ArrayKey):
            del self.arrays[key]

        elif isinstance(key, PointsKey):
            del self.points[key]

        else:
            raise RuntimeError(
                "Only ArrayKey or PointsKey can be used as keys in a "
                "%s."%type(self).__name__)

    def items(self):
        '''Provides a generator iterating over key/value pairs.'''

        for (k, v) in self.arrays.items():
            yield k, v
        for (k, v) in self.points.items():
            yield k, v

    def get_total_roi(self):
        '''Get the union of all the array ROIs in the batch.'''

        total_roi = None

        for collection_type in [self.arrays, self.points]:
            for (key, obj) in collection_type.items():
                if total_roi is None:
                    total_roi = obj.spec.roi
                else:
                    total_roi = total_roi.union(obj.spec.roi)

        return total_roi

    def __repr__(self):

        r = ""
        for collection_type in [self.arrays, self.points]:
            for (key, obj) in collection_type.items():
                r += "%s: %s\n"%(key, obj.spec)
        return r
