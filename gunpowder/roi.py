import copy
from .coordinate import Coordinate
from .freezable import Freezable
import numbers
import numpy as np

class Roi(Freezable):
    '''A rectangular region of interest, defined by an offset and a shape.

    Similar to :class:`Coordinate`, supports simple arithmetics, e.g.::

        roi = Roi((1, 1, 1), (10, 10, 10))
        voxel_size = Coordinate((10, 5, 1))
        scale_shift = roi*voxel_size + 1 # == Roi((11, 6, 2), (101, 51, 11))

    Args:

        offset (array-like of ``int``, optional):

            The starting point (inclusive) of the ROI. Can be ``None``
            (default) if the ROI only characterizes a shape.

        shape (array-like):

            The shape of the ROI. Entries can be ``None`` to indicate
            unboundedness. If ``None`` is passed instead of a tuple, all
            dimensions are set to ``None``, if the number of dimensions can be
            inferred from ``offset``.
    '''

    def __init__(self, offset=None, shape=None):

        self.__offset = None
        self.__shape = None
        self.freeze()

        self.set_shape(shape)
        if offset is not None:
            self.set_offset(offset)

    def set_offset(self, offset):

        self.__offset = Coordinate(offset)
        self.__consolidate_offset()

    def set_shape(self, shape):
        '''Set the shape of this ROI.

        Args:

            shape (array-like or ``None``):

                The new shape. Entries can be ``None`` to indicate
                unboundedness. If ``None`` is passed instead of a tuple, all
                dimensions are set to ``None``, if the number of dimensions can
                be inferred from an existing offset or previous shape.
        '''

        if shape is None:

            if self.__shape is not None:

                dims = self.__shape.dims()

            else:

                assert self.__offset is not None, (
                    "Can not infer dimension of ROI (there is no offset or "
                    "previous shape). Call set_shape with a tuple.")

                dims = self.__offset.dims()

            self.__shape = Coordinate((None,)*dims)

        else:

            self.__shape = Coordinate(shape)

        self.__consolidate_offset()

    def __consolidate_offset(self):
        '''Ensure that offsets for unbound dimensions are None.'''

        if self.__offset is not None:

            assert self.__offset.dims() == self.__shape.dims(), (
                "offset dimension %d != shape dimension %d"%(
                    self.__offset.dims(),
                    self.__shape.dims()))

            self.__offset = Coordinate((
                o
                if s is not None else None
                for o, s in zip(self.__offset, self.__shape)))

    def get_offset(self):
        return self.__offset

    def get_begin(self):
        '''Smallest coordinate inside ROI.'''
        return self.__offset

    def get_end(self):
        '''Smallest coordinate which is component-wise larger than any inside ROI.'''
        if not self.__shape:
            return self.__offset

        return self.__offset + self.__shape

    def get_shape(self):
        return self.__shape

    def get_center(self):

        return self.__offset + self.__shape/2

    def to_slices(self):
        '''Get a ``tuple`` of ``slice`` that represent this ROI and can be used
        to index arrays.'''

        if self.__offset is None:
            return None

        return tuple(
                slice(
                    int(self.__offset[d])
                    if self.__shape[d] is not None
                    else None,
                    int(self.__offset[d] + self.__shape[d])
                    if self.__shape[d] is not None
                    else None)
                for d in range(self.dims())
        )

    def get_bounding_box(self):
        return self.to_slices()

    def dims(self):
        '''The the number of dimensions of this ROI.'''

        if self.__shape is None:
            return 0
        return self.__shape.dims()

    def size(self):
        '''Get the volume of this ROI. Returns ``None`` if the ROI is
        unbounded.'''

        if self.unbounded():
            return None

        size = 1
        for d in self.__shape:
            size *= d
        return size

    def empty(self):
        '''Test if this ROI is empty.'''

        return self.size() == 0

    def unbounded(self):
        '''Test if this ROI is unbounded.'''

        return None in self.__shape

    def contains(self, other):
        '''Test if this ROI contains ``other``, which can be another
        :class:`Roi` or a :class:`Coordinate`.'''

        if isinstance(other, Roi):

            if other.empty():
                return True

            return (
                self.contains(other.get_begin())
                and
                self.contains(other.get_end() - (1,)*other.dims()))

        return all([
            (b is None or p is not None and p >= b)
            and
            (e is None or p is not None and p < e)
            for p, b, e in zip(other, self.get_begin(), self.get_end() )
        ])

    def intersects(self, other):
        '''Test if this ROI intersects with another :class:`Roi`.'''

        assert self.dims() == other.dims()

        if self.empty() or other.empty():
            return False

        # separated if at least one dimension is separated
        separated = any([
            # a dimension is separated if:
            # none of the shapes is unbounded
            (None not in [b1, b2, e1, e2])
            and
            (
                # either b1 starts after e2
                (b1 >= e2)
                or
                # or b2 starts after e1
                (b2 >= e1)
            )
            for b1, b2, e1, e2 in zip(
                self.get_begin(),
                other.get_begin(),
                self.get_end(),
                other.get_end())
        ])

        return not separated

    def intersect(self, other):
        '''Get the intersection of this ROI with another :class:`Roi`.'''

        if not self.intersects(other):
            return Roi(shape=(0,)*self.dims()) # empty ROI

        begin = Coordinate((
            self.__left_max(b1, b2)
            for b1, b2 in zip(self.get_begin(), other.get_begin())
        ))
        end = Coordinate((
            self.__right_min(e1, e2)
            for e1, e2 in zip(self.get_end(), other.get_end())
        ))

        return Roi(begin, end - begin)

    def union(self, other):
        '''Get the union of this ROI with another :class:`Roi`.'''

        begin = Coordinate((
            self.__left_min(b1, b2)
            for b1, b2 in zip(self.get_begin(), other.get_begin())
        ))
        end = Coordinate((
            self.__right_max(e1, e2)
            for e1, e2 in zip(self.get_end(), other.get_end())
        ))

        return Roi(begin, end - begin)

    def shift(self, by):
        '''Shift this ROI.'''

        return Roi(self.__offset + by, self.__shape)

    def snap_to_grid(self, voxel_size, mode='grow'):
        '''Align a ROI with a given voxel size.

        Args:

            voxel_size (:class:`Coordinate`):

                The voxel size of the grid to snap to.

            mode (string, optional):

                How to align the ROI if it is not a multiple of the voxel size.
                Available modes are 'grow', 'shrink', and 'closest'. Defaults to
                'grow'.
        '''

        begin_in_voxel_fractions = (
            np.asarray(self.get_begin(), dtype=np.float32)/
            np.asarray(voxel_size))
        end_in_voxel_fractions = (
            np.asarray(self.get_end(), dtype=np.float32)/
            np.asarray(voxel_size))

        if mode == 'closest':
            begin_in_voxel = np.round(begin_in_voxel_fractions)
            end_in_voxel = np.round(end_in_voxel_fractions)
        elif mode == 'grow':
            begin_in_voxel = np.floor(begin_in_voxel_fractions)
            end_in_voxel = np.ceil(end_in_voxel_fractions)
        elif mode == 'shrink':
            begin_in_voxel = np.ceil(begin_in_voxel_fractions)
            end_in_voxel = np.floor(end_in_voxel_fractions)
        else:
            assert False, 'Unknown mode %s for snap_to_grid'%mode

        return Roi(
            begin_in_voxel*voxel_size,
            (end_in_voxel - begin_in_voxel)*voxel_size)

    def grow(self, amount_neg, amount_pos):
        '''Grow a ROI by the given amounts in each direction:

        Args:

            amount_neg (:class:`Coordinate` or ``None``):

                Amount (per dimension) to grow into the negative direction.

            amount_pos (:class:`Coordinate` or ``None``):

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

    def __left_min(self, x, y):

        # None is considered -inf

        if x is None or y is None:
            return None
        return min(x, y)

    def __left_max(self, x, y):

        # None is considered -inf

        if x is None:
            return y
        if y is None:
            return x
        return max(x, y)

    def __right_min(self, x, y):

        # None is considered +inf

        if x is None:
            return y
        if y is None:
            return x
        return min(x, y)

    def __right_max(self, x, y):

        # None is considered +inf

        if x is None or y is None:
            return None
        return max(x, y)

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
        if self.empty():
            return "[empty ROI]"
        slices = ", ".join(
            [
                (str(b) if b is not None else "") +
                ":" +
                (str(e) if e is not None else "")
                for b, e in zip(self.get_begin(), self.get_end())
            ])
        dims = ", ".join(
            str(a) if a is not None else "inf"
            for a in self.__shape
        )
        return "[" + slices + "] (" + dims + ")"
