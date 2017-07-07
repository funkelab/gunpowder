import numpy as np
import time

from .freezable import Freezable

class Timing(Freezable):

    def __init__(self, node, method_name=''):
        self.__name = type(node).__name__
        self.__method_name = method_name
        self.__start = 0
        self.__time = 0
        self.freeze()

    def start(self):
        self.__start = time.time()

    def stop(self):
        if self.__start == 0:
            return
        self.__time += (time.time() - self.__start)
        self.__start = 0

    def elapsed(self):

        if self.__start == 0:
            return self.__time

        return self.__time + (time.time() - self.__start)

    def get_node_name(self):
        return self.__name

    def get_method_name(self):
        return self.__method_name

class ProfilingStats(Freezable):

    def __init__(self):
        self.__timings = {}
        self.freeze()

    def add(self, timing):
        '''Add a Timing instance. Timings are grouped by their class and method names.'''

        node_name = timing.get_node_name()
        method_name = timing.get_method_name()
        id = (node_name, method_name)

        if id not in self.__timings:
            self.__timings[id] = []
        self.__timings[id].append(timing)

    def merge_with(self, other):
        '''Combine all Timings of two ProfilingStats.'''

        for _, timings in other.__timings.items():
            for timing in timings:
                self.add(timing)

    def __repr__(self):

        rep = ""

        header = ""
        header += "NODE".ljust(20)
        header += "METHOD".ljust(10)
        header += "COUNTS".ljust(10)
        header += "MIN".ljust(10)
        header += "MAX".ljust(10)
        header += "MEAN".ljust(10)
        header += "MEDIAN".ljust(10)
        header += "\n"
        rep += header

        for (node_name, method_name), timings in self.__timings.items():

            times = np.array([ t.elapsed() for t in timings ])
            row = ""
            row += node_name[:19].ljust(20)
            row += method_name[:19].ljust(10)
            row += ("%d"%len(times))[:9].ljust(10)
            row += ("%.2f"%np.min(times))[:9].ljust(10)
            row += ("%.2f"%np.max(times))[:9].ljust(10)
            row += ("%.2f"%np.mean(times))[:9].ljust(10)
            row += ("%.2f"%np.median(times))[:9].ljust(10)
            row += "\n"
            rep += row

        return rep
