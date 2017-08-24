import copy
from .coordinate import Coordinate
from .freezable import Freezable

class VolumeSpec(Freezable):
    '''Contains meta-information about a volume. This is used by 
    :class:`BatchProvider`s to communicate the volumes they offer, as well as by 
    :class:`Volume`s to describe the data they contain.

    Attributes:

        roi (:class:`Roi`): The region of interested represented by this volume 
            spec. Can be `None` for `BatchProvider`s that allow requests for 
            volumes everywhere, but will always be set for volume specs that are 
            part of a :class:`Volume`.

        voxel_size (Coordinate): The size of the spatial axises in world units.

        interpolatable (bool): Whether the values of this volume can be interpolated.

        dtype (np.dtype): The data type of the volume.
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
