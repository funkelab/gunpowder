import copy
from .coordinate import Coordinate
from .freezable import Freezable

class ArraySpec(Freezable):
    '''Contains meta-information about an array. This is used by
    :class:`BatchProviders<BatchProvider>` to communicate the arrays they
    offer, as well as by :class:`Arrays<Array>` to describe the data they
    contain.

    Attributes:

        roi (:class:`Roi`):

            The region of interested represented by this array spec. Can be
            ``None`` for :class:`BatchProviders<BatchProvider>` that allow
            requests for arrays everywhere, but will always be set for array
            specs that are part of a :class:`Array`.

        voxel_size (:class:`Coordinate`):

            The size of the spatial axises in world units.

        interpolatable (``bool``):

            Whether the values of this array can be interpolated.

        dtype (``np.dtype``):

            The data type of the array.
    '''

    def __init__(self, roi=None, voxel_size=None, interpolatable=None, dtype=None):

        self.roi = roi
        self.voxel_size = None if voxel_size is None else Coordinate(voxel_size)
        self.interpolatable = interpolatable
        self.dtype = dtype

        self.freeze()

    def copy(self):
        '''Create a copy of this spec.'''
        return copy.deepcopy(self)

    def __eq__(self, other):

        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):

        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __repr__(self):
        r = ""
        r += "ROI: " + str(self.roi) + ", "
        r += "voxel size: " + str(self.voxel_size) + ", "
        r += "interpolatable: " + str(self.interpolatable) + ", "
        r += "dtype: " + str(self.dtype)
        return r
