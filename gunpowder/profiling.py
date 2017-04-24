from freezable import Freezable
import time

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

    def __repr__(self):
        return self.__name + ": " + str(self.__time)

class ProfilingStats(Freezable):

    def __init__(self):
        self.__timings = []
        self.freeze()

    def add(self, timing):
        self.__timings.append(timing)

    def __repr__(self):
        rep = ""
        for t in self.__timings:
            rep += str(t) + "\n"
        return rep
