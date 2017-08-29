import copy
import numpy as np
import time

from .freezable import Freezable

class Timing(Freezable):

    def __init__(self, node, method_name=None):
        self.__name = type(node).__name__
        self.__method_name = method_name
        self.__start = 0
        self.__first_start = 0
        self.__last_stop = 0
        self.__time = 0
        self.freeze()

    def start(self):
        self.__start = time.time()
        if self.__first_start == 0:
            self.__first_start = self.__start

    def stop(self):
        if self.__start == 0:
            return
        t = time.time()
        self.__time += (t - self.__start)
        self.__start = 0
        self.__last_stop = t

    def elapsed(self):
        '''Accumulated time elapsed between calls to start() and stop().'''

        if self.__start == 0:
            return self.__time

        return self.__time + (time.time() - self.__start)

    def span(self):
        '''Timestamps of the first call to start() and last call to stop().'''
        return self.__first_start, self.__last_stop

    def get_node_name(self):
        return self.__name

    def get_method_name(self):
        return self.__method_name

class TimingSummary(Freezable):
    '''Holds repeated Timings of the same node/method to be queried for statistics.'''

    def __init__(self):
        self.timings = []
        self.times = []
        self.freeze()

    def add(self, timing):
        '''Add a Timing to this summary.'''
        self.timings.append(timing)
        self.times.append(timing.elapsed())

    def merge(self, other):
        '''Merge another summary into this one.'''
        for timing in other.timings:
            self.add(timing)

    def counts(self):
        return len(self.times)

    def min(self):
        return np.min(self.times)

    def max(self):
        return np.max(self.times)

    def mean(self):
        return np.mean(self.times)

    def median(self):
        return np.median(self.times)

class ProfilingStats(Freezable):

    def __init__(self):
        self.__summaries = {}
        self.freeze()

    def add(self, timing):
        '''Add a Timing instance. Timings are grouped by their class and method names.'''

        node_name = timing.get_node_name()
        method_name = timing.get_method_name()
        id = (node_name, method_name)

        if id not in self.__summaries:
            self.__summaries[id] = TimingSummary()
        self.__summaries[id].add(copy.deepcopy(timing))

    def merge_with(self, other):
        '''Combine statitics of two ProfilingStats.'''

        for id, summary in other.__summaries.items():
            if id in self.__summaries:
                self.__summaries[id].merge(copy.deepcopy(summary))
            else:
                self.__summaries[id] = copy.deepcopy(summary)

    def get_timing_summaries(self):
        '''Get a dictionary (node_name,method_name) -> TimingSummary.'''
        return self.__summaries

    def get_timing_summary(self, node_name, method_name=None):
        '''Get a :class:`TimingSummary` for the given node and method name.'''

        if (node_name, method_name) not in self.__summaries:
            raise RuntimeError("No timing summary for node %s, method %s"%(node_name,method_name))

        return self.__summaries[(node_name,method_name)]

    def span(self):
        '''Timestamps of the first call to start() and last call to stop() over 
        all Timings added.'''

        spans = [t.span() for (_, summary) in self.__summaries.items() for t in summary.timings]
        first_start = min([span[0] for span in spans])
        last_stop = max([span[1] for span in spans])

        return first_start, last_stop

    def span_time(self):
        '''Time between the first call to start() and last call to stop() over 
        any timing.'''

        start, stop = self.span()
        return stop - start
