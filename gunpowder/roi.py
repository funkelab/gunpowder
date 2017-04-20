from freezable import Freezable

class Roi(Freezable):
    '''A rectengular region of interest, defined by an offset and a shape.
    '''

    def __init__(self, offset=None, shape=None):
        self.__offset = None if not offset else tuple(offset)
        self.__shape = None if not shape else tuple(shape)
        self.freeze()

    def set_offset(self, offset):
        self.__offset = tuple(offset)

    def set_shape(self, shape):
        self.__shape = tuple(shape)

    def get_offset(self):
        return self.__offset

    def get_shape(self):
        return self.__shape

    def get_bounding_box(self):

        if self.__offset is None or self.__shape is None:
            return None

        return tuple(
                slice(self.__offset[d], self.__shape[d] + self.__offset[d])
                for d in range(self.dims())
        )

    def dims(self):

        if self.__shape is None:
            return 0
        return len(self.__shape)

    def size(self):

        if self.__shape is None:
            return 0

        size = 1
        for d in self.__shape:
            size *= d
        return size

    def contains(self, other):

        bb1 = self.get_bounding_box()
        bb2 = other.get_bounding_box()
        contained = [
                bb1[d].start <= bb2[d].start and bb1[d].stop >= bb2[d].stop
                for d in range(self.dims())
        ]
        return all(contained)

    def intersects(self, other):

        bb1 = self.get_bounding_box()
        bb2 = other.get_bounding_box()
        separated = [
                bb1[d].start >= bb2[d].stop or bb2[d].start >= bb1[d].stop
                for d in range(self.dims())
        ]
        return not any(separated)

    def intersect(self, other):

        if not self.intersects(other):
            return None

        assert self.dims() == other.dims()

        offset = tuple(
                max(self.__offset[d], other.__offset[d])
                for d in range(self.dims())
        )
        shape = tuple(
                min(self.__offset[d] + self.__shape[d], other.__offset[d] + other.__shape[d]) - offset[d]
                for d in range(self.dims())
        )

        return Roi(offset, shape)

    def shift(self, by):
        offset = tuple(
                self.__offset[d] + by[d]
                for d in range(self.dims())
        )
        return Roi(offset, self.__shape)

    def __repr__(self):
        return str(self.__offset) + "+" + str(self.__shape)
