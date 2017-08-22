from .freezable import Freezable
from gunpowder.points import PointsType
from gunpowder.points_spec import PointsSpec
from gunpowder.volume import VolumeType
from gunpowder.volume_spec import VolumeSpec
import copy

class ProviderSpec(Freezable):
    '''A collection of (possibly partial) :class:`VolumeSpec`s and 
    :class:`PointsSpec`s describing a :class:`BatchProvider`'s offered volumes 
    and points.

    Args:

        volume_specs (dict): A dictionary from :class:`VolumeType` to :class:`VolumeSpec`.

        points_specs (dict): A dictionary from :class:`PointsType` to :class:`PointsSpec`.
    '''

    def __init__(self, volume_specs=None, points_specs=None):

        if volume_specs is None:
            self.volume_specs = {}
        else:
            self.volume_specs = volume_specs

        if points_specs is None:
            self.points_specs = {}
        else:
            self.points_specs = points_specs

        self.freeze()

    def __setitem__(self, identifier, spec):

        if isinstance(spec, VolumeSpec):
            assert isinstance(identifier, VolumeType), "Only a VolumeType is allowed as key for a VolumeSpec value."
            self.volume_specs[identifier] = copy.deepcopy(spec)

        elif isinstance(spec, PointsSpec):
            assert isinstance(identifier, PointsType), "Only a PointsType is allowed as key for a PointsSpec value."
            self.points_specs[identifier] = copy.deepcopy(spec)

        else:
            raise RuntimeError("Only VolumeSpec or PointsSpec can be set in a %s."%type(self).__name__)

    def __getitem__(self, identifier):

        if isinstance(identifier, VolumeType):
            return self.volume_specs[identifier]

        elif isinstance(identifier, PointsType):
            return self.points_specs[identifier]

        else:
            raise RuntimeError("Only VolumeSpec or PointsSpec can be used as keys in a %s."%type(self).__name__)

    def __contains__(self, identifier):

        if isinstance(identifier, VolumeType):
            return identifier in self.volume_specs

        elif isinstance(identifier, PointsType):
            return identifier in self.points_specs

        else:
            raise RuntimeError("Only VolumeSpec or PointsSpec can be used as keys in a %s."%type(self).__name__)

    def __delitem__(self, identifier):

        if isinstance(identifier, VolumeType):
            del self.volume_specs[identifier]

        elif isinstance(identifier, PointsType):
            del self.points_specs[identifier]

        else:
            raise RuntimeError("Only VolumeSpec or PointsSpec can be used as keys in a %s."%type(self).__name__)

    def items(self):

        for (k,v) in self.volume_specs.items():
            yield k,v
        for (k,v) in self.points_specs.items():
            yield k,v

    def get_total_roi(self):
        '''Get the union of all the ROIs.'''

        total_roi = None
        for specs_type in [self.volume_specs, self.points_specs]:
            for (type, spec) in specs_type.items():
                if total_roi is None:
                    total_roi = spec.roi
                else:
                    total_roi = total_roi.union(spec.roi)
            return total_roi

    def get_common_roi(self):
        ''' Get the intersection of all the requested ROIs.'''

        common_roi = None
        for specs_type in [self.volume_specs, self.points_specs]:
            for (type, spec) in specs_type.items():
                if common_roi is None:
                    common_roi = spec.roi
                else:
                    common_roi = common_roi.intersect(spec.roi)

        return common_roi

    def __eq__(self, other):

        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):

        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __repr__(self):

        r = "\n"
        for specs_type in [self.volume_specs, self.points_specs]:
            for (type, spec) in specs_type.items():
                r += "\t%s: %s\n"%(type, spec)
        return r
