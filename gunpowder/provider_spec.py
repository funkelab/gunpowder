import math
from gunpowder.coordinate import Coordinate
from gunpowder.array import ArrayKey
from gunpowder.array_spec import ArraySpec
from gunpowder.graph import GraphKey
from gunpowder.graph_spec import GraphSpec
from gunpowder.roi import Roi
from .freezable import Freezable
import time
import logging
import copy

logger = logging.getLogger(__name__)


import logging
import warnings

logger = logging.getLogger(__file__)

class ProviderSpec(Freezable):
    '''A collection of (possibly partial) :class:`ArraySpecs<ArraySpec>` and
    :class:`GraphSpecs<GraphSpec>` describing a
    :class:`BatchProvider's<BatchProvider>` offered arrays and graphs.

    This collection mimics a dictionary. Specs can be added with::

        provider_spec = ProviderSpec()
        provider_spec[array_key] = ArraySpec(...)
        provider_spec[graph_key] = GraphSpec(...)

    Here, ``array_key`` and ``graph_key`` are :class:`ArrayKey` and
    :class:`GraphKey`. The specs can be queried with::

        array_spec = provider_spec[array_key]
        graph_spec = provider_spec[graph_key]

    Furthermore, pairs of keys/values can be iterated over using
    ``provider_spec.items()``.

    To access only array or graph specs, use the dictionaries
    ``provider_spec.array_specs`` or ``provider_spec.graph_specs``,
    respectively.

    Args:

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`):

            Initial array specs.

        graph_specs (``dict``, :class:`GraphKey` -> :class:`GraphSpec`):

            Initial graph specs.

    Attributes:

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`):

            Contains all array specs contained in this provider spec.

        graph_specs (``dict``, :class:`GraphKey` -> :class:`GraphSpec`):

            Contains all graph specs contained in this provider spec.
    '''

    def __init__(self, array_specs=None, graph_specs=None, points_specs=None):

        self.array_specs = {}
        self.graph_specs = {}
        self.freeze()

        # use __setitem__ instead of copying the dicts, this ensures type tests
        # are run
        if array_specs is not None:
            for key, spec in array_specs.items():
                self[key] = spec
        if graph_specs is not None:
            for key, spec in graph_specs.items():
                self[key] = spec
        if points_specs is not None:
            for key, spec in points_specs.items():
                self[key] = spec

    @property
    def points_specs(self):
        # Alias to graphs
        warnings.warn(
            "points_specs are depricated. Please use graph_specs", DeprecationWarning
        )
        return self.graph_specs

    def __setitem__(self, key, spec):

        assert isinstance(key, ArrayKey) or isinstance(key, GraphKey), \
            f"Only ArrayKey or GraphKey (not {type(key).__name__} are " \
            "allowed as key for ProviderSpec, "

        if isinstance(key, ArrayKey):

            if isinstance(spec, Roi):
                spec = ArraySpec(roi=spec)

            assert isinstance(spec, ArraySpec), \
                f"Only ArraySpec (not {type(spec).__name__}) can be set for " \
                "ArrayKey"

            self.array_specs[key] = spec.copy()

        else:

            if isinstance(spec, Roi):
                spec = GraphSpec(roi=spec)

            assert isinstance(spec, GraphSpec), \
                f"Only GraphSpec (not {type(spec).__name__}) can be set for " \
                "GraphKey"

            self.graph_specs[key] = spec.copy()

    def __getitem__(self, key):

        if isinstance(key, ArrayKey):
            return self.array_specs[key]

        elif isinstance(key, GraphKey):
            return self.graph_specs[key]
        else:
            raise RuntimeError(
                "Only ArrayKey or GraphKey can be used as keys in a "
                "%s."%type(self).__name__)

    def __len__(self):

        return len(self.array_specs) + len(self.graph_specs)

    def __contains__(self, key):

        if isinstance(key, ArrayKey):
            return key in self.array_specs

        elif isinstance(key, GraphKey):
            return key in self.graph_specs

        else:
            raise RuntimeError(
                "Only ArrayKey or GraphKey, can be used as keys in a "
                "%s. Key %s is a %s"%(type(self).__name__, key, type(key).__name__))

    def __delitem__(self, key):

        if isinstance(key, ArrayKey):
            del self.array_specs[key]

        elif isinstance(key, GraphKey):
            del self.graph_specs[key]

        else:
            raise RuntimeError(
                "Only ArrayKey or GraphKey can be used as keys in a "
                "%s."%type(self).__name__)

    def remove_placeholders(self):
        self.array_specs = {k: v for k, v in self.array_specs.items() if not v.placeholder}
        self.graph_specs = {k: v for k, v in self.graph_specs.items() if not v.placeholder}

    def items(self):
        '''Provides a generator iterating over key/value pairs.'''

        for (k, v) in self.array_specs.items():
            yield k, v
        for (k, v) in self.graph_specs.items():
            yield k, v

    def get_total_roi(self):
        '''Get the union of all the ROIs.'''

        total_roi = None
        for specs_type in [self.array_specs, self.graph_specs]:
            for (_, spec) in specs_type.items():
                if total_roi is None:
                    total_roi = spec.roi
                elif spec.roi is not None:
                    total_roi = total_roi.union(spec.roi)
        return total_roi

    def get_common_roi(self):
        '''Get the intersection of all the requested ROIs.'''

        common_roi = None
        for specs_type in [self.array_specs, self.graph_specs]:
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

        if array_keys is None:
            array_keys = self.array_specs.keys()

        if not array_keys:
            return None

        lcm_voxel_size = None
        for key in array_keys:
            voxel_size = self.array_specs[key].voxel_size
            if voxel_size is None:
                continue
            if lcm_voxel_size is None:
                lcm_voxel_size = voxel_size
            else:
                lcm_voxel_size = Coordinate(
                    (a * b // math.gcd(a, b)
                     for a, b in zip(lcm_voxel_size, voxel_size)))

        return lcm_voxel_size

    def __eq__(self, other):

        if isinstance(other, self.__class__):
            other_dict = copy.deepcopy(other.__dict__)
            self_dict = copy.deepcopy(self.__dict__)
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
