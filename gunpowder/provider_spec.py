import fractions
from gunpowder.coordinate import Coordinate
from gunpowder.points import PointsKey
from gunpowder.points_spec import PointsSpec
from gunpowder.array import ArrayKey
from gunpowder.array_spec import ArraySpec
from .freezable import Freezable

class ProviderSpec(Freezable):
    '''A collection of (possibly partial) :class:`ArraySpec`s and
    :class:`PointsSpec`s describing a :class:`BatchProvider`'s offered arrays
    and points.

    This collection mimics a dictionary. Specs can be added with::

        provider_spec = ProviderSpec()
        provider_spec[array_type] = ArraySpec(...)
        provider_spec[points_type] = PointsSpec(...)

    Here, ``array_type`` and ``points_type`` are :class:`ArrayKey` and
    :class:`PointsKey` instances, previously registered with
    :fun:`register_array_type` or :fun:`register_points_type`. The specs can
    be queried with::

        array_spec = provider_spec[array_type]
        points_spec = provider_spec[points_type]

    Furthermore, pairs of keys/values can be iterated over using
    ``provider_spec.items()``.

    To access only array or points specs, use the dictionaries
    ``provider_spec.array_specs`` or ``provider_spec.points_specs``,
    respectively.

    Args:

        array_specs (dict): A dictionary from :class:`ArrayKey` to
            :class:`ArraySpec`.

        points_specs (dict): A dictionary from :class:`PointsKey` to
            :class:`PointsSpec`.
    '''

    def __init__(self, array_specs=None, points_specs=None):

        self.array_specs = {}
        self.points_specs = {}
        self.freeze()

        # use __setitem__ instead of copying the dicts, this ensures type tests
        # are run
        if array_specs is not None:
            for identifier, spec in array_specs.items():
                self[identifier] = spec
        if points_specs is not None:
            for identifier, spec in points_specs.items():
                self[identifier] = spec


    def __setitem__(self, identifier, spec):

        if isinstance(spec, ArraySpec):
            assert isinstance(identifier, ArrayKey), ("Only a ArrayKey is "
                                                        "allowed as key for a "
                                                        "ArraySpec value.")
            self.array_specs[identifier] = spec.copy()

        elif isinstance(spec, PointsSpec):
            assert isinstance(identifier, PointsKey), ("Only a PointsKey is "
                                                        "allowed as key for a "
                                                        "PointsSpec value.")
            self.points_specs[identifier] = spec.copy()

        else:
            raise RuntimeError("Only ArraySpec or PointsSpec can be set in a "
                               "%s."%type(self).__name__)

    def __getitem__(self, identifier):

        if isinstance(identifier, ArrayKey):
            return self.array_specs[identifier]

        elif isinstance(identifier, PointsKey):
            return self.points_specs[identifier]

        else:
            raise RuntimeError("Only ArraySpec or PointsSpec can be used as "
                               "keys in a %s."%type(self).__name__)

    def __len__(self):

        return len(self.array_specs) + len(self.points_specs)

    def __contains__(self, identifier):

        if isinstance(identifier, ArrayKey):
            return identifier in self.array_specs

        elif isinstance(identifier, PointsKey):
            return identifier in self.points_specs

        else:
            raise RuntimeError("Only ArraySpec or PointsSpec can be used as "
                               "keys in a %s."%type(self).__name__)

    def __delitem__(self, identifier):

        if isinstance(identifier, ArrayKey):
            del self.array_specs[identifier]

        elif isinstance(identifier, PointsKey):
            del self.points_specs[identifier]

        else:
            raise RuntimeError("Only ArraySpec or PointsSpec can be used as "
                               "keys in a %s."%type(self).__name__)

    def items(self):
        '''Provides a generator iterating over key/value pairs.'''

        for (k, v) in self.array_specs.items():
            yield k, v
        for (k, v) in self.points_specs.items():
            yield k, v

    def get_total_roi(self):
        '''Get the union of all the ROIs.'''

        total_roi = None
        for specs_type in [self.array_specs, self.points_specs]:
            for (_, spec) in specs_type.items():
                if total_roi is None:
                    total_roi = spec.roi
                else:
                    total_roi = total_roi.union(spec.roi)
            return total_roi

    def get_common_roi(self):
        '''Get the intersection of all the requested ROIs.'''

        common_roi = None
        for specs_type in [self.array_specs, self.points_specs]:
            for (_, spec) in specs_type.items():
                if common_roi is None:
                    common_roi = spec.roi
                else:
                    common_roi = common_roi.intersect(spec.roi)

        return common_roi

    def get_lcm_voxel_size(self, array_types=None):
        '''Get the least common multiple of the voxel sizes in this spec.

        Args:

            array_types (list of :class:`ArrayKey`, optional): If given,
                consider only the given array types.
        '''

        if array_types is None:
            array_types = self.array_specs.keys()

        if not array_types:
            raise RuntimeError("Can not compute lcm voxel size -- there are "
                               "no array specs in this provider spec.")
        else:
            if not array_types:
                raise RuntimeError("Can not compute lcm voxel size -- list of "
                                   "given array specs is empty.")

        lcm_voxel_size = None
        for identifier in array_types:
            voxel_size = self.array_specs[identifier].voxel_size
            if lcm_voxel_size is None:
                lcm_voxel_size = voxel_size
            else:
                lcm_voxel_size = Coordinate(
                    (a * b // fractions.gcd(a, b)
                     for a, b in zip(lcm_voxel_size, voxel_size)))

        return lcm_voxel_size

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
        for (identifier, spec) in self.items():
            r += "\t%s: %s\n"%(identifier, spec)
        return r
