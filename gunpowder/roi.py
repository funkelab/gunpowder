import copy
from .coordinate import Coordinate
from .freezable import Freezable
import numbers

class Roi(Freezable):
    '''A rectangular region of interest, defined by an offset and a shape.

    Args:

        offset (array-like of int, optional): The starting point (inclusive) of
            the ROI. Can be `None` (default) if the ROI only characterizes a
            shape.

        shape (array-like of int, optional): The shape of the ROI. Can be
            `None` (default) to create an empty ROI.
    '''

    def __init__(self, offset=None, shape=None):
        self.__offset = None if offset is None else Coordinate(offset)
        self.__shape = None if shape is None else Coordinate(shape)
        self.freeze()

        if self.__offset is not None and self.__shape is not None:
            assert self.__offset.dims() == self.__shape.dims(), (
                "offset dimension %d != shape dimension %d"%(
                    self.__offset.dims(),
                    self.__shape.dims()))

    def set_offset(self, offset):
        self.__offset = Coordinate(offset)
        if self.__shape is not None:
            assert self.__offset.dims() == self.__shape.dims(), (
                "offset dimension %d != shape dimension %d"%(
                    self.__offset.dims(),
                    self.__shape.dims()))

    def set_shape(self, shape):
        self.__shape = Coordinate(shape)
        if self.__offset is not None:
            assert self.__offset.dims() == self.__shape.dims(), (
                "offset dimension %d != shape dimension %d"%(
                    self.__offset.dims(),
                    self.__shape.dims()))

    def get_offset(self):
        return self.__offset

    def get_begin(self):
        '''Smallest coordinate inside ROI.'''
        return self.__offset

    def get_end(self):
        '''Smallest coordinate which is component-wise larger than any inside ROI.'''
        return self.__offset + self.__shape

    def get_shape(self):
        return self.__shape

    def get_center(self):

        return self.__offset + self.__shape/2

    def get_bounding_box(self):

        if self.__offset is None or self.__shape is None:
            return None

        return tuple(
                slice(int(self.__offset[d]), int(self.__shape[d] + self.__offset[d]))
                for d in range(self.dims())
        )

    def dims(self):

        if self.__shape is None:
            return 0
        return self.__shape.dims()

    def size(self):

        if self.__shape is None:
            return 0

        size = 1
        for d in self.__shape:
            size *= d
        return size

    def empty(self):

        return self.size() == 0

    def contains(self, other):

        if isinstance(other, Roi):

            bb1 = self.get_bounding_box()
            bb2 = other.get_bounding_box()
            contained = [
                    bb1[d].start <= bb2[d].start and bb1[d].stop >= bb2[d].stop
                    for d in range(self.dims())
            ]
            return all(contained)

        elif isinstance(other, Coordinate):

            return all([ p >= b and p < e for p, b, e in zip(other, self.get_begin(), self.get_end() )])

        else:

            raise RuntimeError("contains() can only be applied to Roi and Coordinate")

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
            return Roi() # empty ROI

        assert self.dims() == other.dims()

        offset = Coordinate(
                max(self.__offset[d], other.__offset[d])
                for d in range(self.dims())
        )
        shape = Coordinate(
                min(self.__offset[d] + self.__shape[d], other.__offset[d] + other.__shape[d]) - offset[d]
                for d in range(self.dims())
        )

        return Roi(offset, shape)

    def union(self, other):

        assert self.dims() == other.dims(), "Can not compute union of ROI with dim %d and %d"%(self.dims(), other.dims())

        offset = Coordinate(
                min(self.__offset[d], other.__offset[d])
                for d in range(self.dims())
        )
        shape = Coordinate(
                max(self.__offset[d] + self.__shape[d], other.__offset[d] + other.__shape[d]) - offset[d]
                for d in range(self.dims())
        )

        return Roi(offset, shape)

    def shift(self, by):

        return Roi(self.__offset + by, self.__shape)

    def grow(self, amount_neg, amount_pos):
        '''Grow a ROI by the given amounts in each direction:

        amount_neg: Coordinate or None

            Amount (per dimension) to grow into the negative direction.

        amount_pos: Coordinate or None

            Amount (per dimension) to grow into the positive direction.
        '''

        if amount_neg is None:
            amount_neg = Coordinate((0,)*self.dims())
        if amount_pos is None:
            amount_pos = Coordinate((0,)*self.dims())

        assert len(amount_neg) == self.dims()
        assert len(amount_pos) == self.dims()

        offset = self.__offset - amount_neg
        shape = self.__shape + amount_neg + amount_pos

        return Roi(offset, shape)

    def copy(self):
        '''Create a copy of this ROI.'''
        return copy.deepcopy(self)

    def __add__(self, other):

        assert isinstance(other, tuple), "can only add Coordinate or tuples to Roi"
        return self.shift(other)

    def __sub__(self, other):

        assert isinstance(other, Coordinate), "can only subtract Coordinate from Roi"
        return self.shift(-other)

    def __mul__(self, other):

        assert isinstance(other, tuple) or isinstance(other, numbers.Number), "can only multiply with a number or tuple of numbers"
        return Roi(self.__offset*other, self.__shape*other)

    def __div__(self, other):

        assert isinstance(other, tuple) or isinstance(other, numbers.Number), "can only divide by a number or tuple of numbers"
        return Roi(self.__offset/other, self.__shape/other)

    def __truediv__(self, other):

        assert isinstance(other, tuple) or isinstance(other, numbers.Number), "can only divide by a number or tuple of numbers"
        return Roi(self.__offset/other, self.__shape/other)

    def __floordiv__(self, other):

        assert isinstance(other, tuple) or isinstance(other, numbers.Number), "can only divide by a number or tuple of numbers"
        return Roi(self.__offset//other, self.__shape//other)

    def __eq__(self, other):

        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):

        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __repr__(self):
        return str(self.get_begin()) + "--" + str(self.get_end()) + " [" + "x".join(str(a) for a in self.__shape) + "]"
