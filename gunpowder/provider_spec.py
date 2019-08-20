import math
from gunpowder.coordinate import Coordinate
from gunpowder.points import PointsKey
from gunpowder.points_spec import PointsSpec
from gunpowder.array import ArrayKey
from gunpowder.array_spec import ArraySpec
from .freezable import Freezable
import time
import logging
import copy

logger = logging.getLogger(__name__)


class ProviderSpec(Freezable):
    '''A collection of (possibly partial) :class:`ArraySpecs<ArraySpec>` and
    :class:`PointsSpecs<PointsSpec>` describing a
    :class:`BatchProvider's<BatchProvider>` offered arrays and points.

    This collection mimics a dictionary. Specs can be added with::

        provider_spec = ProviderSpec()
        provider_spec[array_key] = ArraySpec(...)
        provider_spec[points_key] = PointsSpec(...)

    Here, ``array_key`` and ``points_key`` are :class:`ArrayKey` and
    :class:`PointsKey`. The specs can be queried with::

        array_spec = provider_spec[array_key]
        points_spec = provider_spec[points_key]

    Furthermore, pairs of keys/values can be iterated over using
    ``provider_spec.items()``.

    To access only array or points specs, use the dictionaries
    ``provider_spec.array_specs`` or ``provider_spec.points_specs``,
    respectively.

    Args:

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`):

            Initial array specs.

        points_specs (``dict``, :class:`PointsKey` -> :class:`PointsSpec`):

            Initial points specs.

        random_seed (``int``, optional):

            A random seed to use for this request. Makes sure
            that requests can be repeated.

    Attributes:

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`):

            Contains all array specs contained in this provider spec.

        points_specs (``dict``, :class:`PointsKey` -> :class:`PointsSpec`):

            Contains all points specs contained in this provider spec.
        
        place_holders (``dict``, :class:`PointsKey` -> :class: `PointsSpec` or `ArrayKey` -> :class:`ArraySpec`)

            Contains all placeholders. Used only for checking request consistency
            and calculating request lcm voxel size and rois.
    '''

    def __init__(self, array_specs=None, points_specs=None, random_seed: int = None):

        self.array_specs = {}
        self.points_specs = {}
        self.place_holders = {}
        self._random_seed = (
            random_seed if random_seed is not None else int(time.time() * 1e6)
        )
        self.freeze()

        # use __setitem__ instead of copying the dicts, this ensures type tests
        # are run
        if array_specs is not None:
            for key, spec in array_specs.items():
                self[key] = spec
        if points_specs is not None:
            for key, spec in points_specs.items():
                self[key] = spec

    @property
    def random_seed(self):
        return self._random_seed % (2 ** 32)

    def __setitem__(self, key, spec):

        if isinstance(spec, ArraySpec):
            assert isinstance(key, ArrayKey), ("Only a ArrayKey is "
                                                        "allowed as key for a "
                                                        "ArraySpec value.")
            self.array_specs[key] = spec.copy()

        elif isinstance(spec, PointsSpec):
            assert isinstance(key, PointsKey), ("Only a PointsKey is "
                                                        "allowed as key for a "
                                                        "PointsSpec value.")
            self.points_specs[key] = spec.copy()

        else:
            raise RuntimeError("Only ArraySpec or PointsSpec can be set in a "
                               "%s."%type(self).__name__)

    def __getitem__(self, key):

        if isinstance(key, ArrayKey):
            return self.array_specs[key]

        elif isinstance(key, PointsKey):
            return self.points_specs[key]

        else:
            raise RuntimeError(
                "Only ArrayKey or PointsKey can be used as keys in a "
                "%s."%type(self).__name__)

    def __len__(self):

        return len(self.array_specs) + len(self.points_specs)

    def __contains__(self, key):

        if isinstance(key, ArrayKey):
            return key in self.array_specs

        elif isinstance(key, PointsKey):
            return key in self.points_specs

        else:
            raise RuntimeError(
                "Only ArrayKey or PointsKey can be used as keys in a "
                "%s."%type(self).__name__)

    def __delitem__(self, key):

        if isinstance(key, ArrayKey):
            del self.array_specs[key]

        elif isinstance(key, PointsKey):
            del self.points_specs[key]

        else:
            raise RuntimeError(
                "Only ArrayKey or PointsKey can be used as keys in a "
                "%s."%type(self).__name__)

    def items(self):
        '''Provides a generator iterating over key/value pairs.'''

        for (k, v) in self.array_specs.items():
            yield k, v
        for (k, v) in self.points_specs.items():
            yield k, v

    def get_total_roi(self):
        '''Get the union of all the ROIs.'''

        total_roi = None
        for specs_type in [self.array_specs, self.points_specs, self.place_holders]:
            for (_, spec) in specs_type.items():
                if total_roi is None:
                    total_roi = spec.roi
                elif spec.roi is not None:
                    total_roi = total_roi.union(spec.roi)
        return total_roi

    def get_common_roi(self):
        '''Get the intersection of all the requested ROIs.'''

        common_roi = None
        for specs_type in [self.array_specs, self.points_specs, self.place_holders]:
            for (_, spec) in specs_type.items():
                if common_roi is None:
                    common_roi = spec.roi
                else:
                    common_roi = common_roi.intersect(spec.roi)

        return common_roi

    def get_lcm_voxel_size(self, array_keys=None):
        '''Get the least common multiple of the voxel sizes in this spec.

        Args:

            array_keys (list of :class:`ArrayKey`, optional): If given,
                consider only the given array types.
        '''

        specs = []
        if array_keys is None:
            specs = [spec for spec in self.array_specs.values()] + [
                spec for spec in self.place_holders.values() if isinstance(spec, ArraySpec)
            ]
        else:
            specs = [
                self.array_specs[k] if k in self.array_specs else self.place_holders[k]
                for k in array_keys
            ]

        lcm_voxel_size = Coordinate((1,) * self.get_total_roi().dims())

        if not specs:
            logger.warning(
                (
                    "Can not compute lcm voxel size -- there are "
                    "no array specs in this provider spec.\nAssuming a voxel size of {}"
                ).format(lcm_voxel_size)
            )
            return lcm_voxel_size

        for spec in specs:
            voxel_size = spec.voxel_size
            if voxel_size is None:
                continue
            else:
                lcm_voxel_size = Coordinate(
                    (a * b // math.gcd(a, b) for a, b in zip(lcm_voxel_size, voxel_size))
                )

        return lcm_voxel_size

    def _update_random_seed(self):
        self._random_seed = hash((self._random_seed + 1) ** 2)

    def __eq__(self, other):

        if isinstance(other, self.__class__):
            other_dict = copy.deepcopy(other.__dict__)
            self_dict = copy.deepcopy(self.__dict__)
            other_dict.pop("_random_seed")
            self_dict.pop("_random_seed")
            return self_dict == other_dict
        return NotImplemented

    def __ne__(self, other):

        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __repr__(self):

        r = "\n"
        for (key, spec) in self.items():
            r += "\t%s: %s\n"%(key, spec)
        return r
