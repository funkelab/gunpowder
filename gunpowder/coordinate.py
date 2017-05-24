import numbers

class Coordinate(tuple):

    def __new__(cls, array_like):
        return super(Coordinate, cls).__new__(cls, array_like)

    def dims(self):
        return len(self)

    def __neg__(self):

        return Coordinate(-a for a in self)

    def __abs__(self):

        return Coordinate(abs(a) for a in self)

    def __add__(self, other):

        assert isinstance(other, tuple), "can only add Coordinate or tuples to Coordinate"
        assert self.dims() == len(other), "can only add Coordinate of equal dimensions"

        return Coordinate(a+b for a,b in zip(self, other))

    def __sub__(self, other):

        assert isinstance(other, tuple), "can only subtract Coordinate or tuples to Coordinate"
        assert self.dims() == len(other), "can only subtract Coordinate of equal dimensions"

        return Coordinate(a-b for a,b in zip(self, other))

    def __mul__(self, other):

        if isinstance(other, tuple):

            assert self.dims() == len(other), "can only multiply Coordinate of equal dimensions"

            return Coordinate(a*b for a,b in zip(self, other))

        if isinstance(other, numbers.Number):

            return Coordinate(a*other for a in self)

    def __div__(self, other):

        if isinstance(other, tuple):

            assert self.dims() == len(other), "can only divide Coordinate of equal dimensions"

            return Coordinate(a/b for a,b in zip(self, other))

        if isinstance(other, numbers.Number):

            return Coordinate(a/other for a in self)

    def __truediv__(self, other):

        if isinstance(other, tuple):

            assert self.dims() == len(other), "can only divide Coordinate of equal dimensions"

            return Coordinate(a/b for a,b in zip(self, other))

        if isinstance(other, numbers.Number):

            return Coordinate(a/other for a in self)

    def __floordiv__(self, other):

        if isinstance(other, tuple):

            assert self.dims() == len(other), "can only divide Coordinate of equal dimensions"

            return Coordinate(a//b for a,b in zip(self, other))

        if isinstance(other, numbers.Number):

            return Coordinate(a//other for a in self)
