import numpy as np
import time

from .freezable import Freezable

class Timing(Freezable):

    def __init__(self, instance):
        self.__name = type(instance).__name__
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

    def get_name(self):
        return self.__name

class ProfilingStats(Freezable):

    def __init__(self):
        self.__timings = {}
        self.freeze()

    def add(self, timing):
        '''Add a Timing instance. Timings are grouped by their name'''

        name = timing.get_name()

        if name not in self.__timings:
            self.__timings[name] = []
        self.__timings[name].append(timing)

    def merge_with(self, other):
        '''Combine all Timings of two ProfilingStats.'''

        for name, timings in other.__timings.items():
            for timing in timings:
                self.add(timing)

    def __repr__(self):

        rep = ""

        header = ""
        header += "NODE".ljust(20)
        header += "COUNTS".ljust(10)
        header += "MIN".ljust(10)
        header += "MAX".ljust(10)
        header += "MEAN".ljust(10)
        header += "MEDIAN".ljust(10)
        header += "\n"
        rep += header

        for name, timings in self.__timings.items():

            times = np.array([ t.elapsed() for t in timings ])
            row = ""
            row += name[:19].ljust(20)
            row += ("%d"%len(times))[:9].ljust(10)
            row += ("%.2f"%np.min(times))[:9].ljust(10)
            row += ("%.2f"%np.max(times))[:9].ljust(10)
            row += ("%.2f"%np.mean(times))[:9].ljust(10)
            row += ("%.2f"%np.median(times))[:9].ljust(10)
            row += "\n"
            rep += row

        return rep
