from random import randrange
from random import choice, seed
import logging
import numpy as np

from gunpowder.coordinate import Coordinate
from gunpowder.batch_request import BatchRequest

from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class SpecifiedLocation(BatchFilter):
    '''Choses a batch at a location from the list provided at init, making sure
    it is in the bounding box of the upstream provider.

    Locations should be given in world units.

    Locations will be chosen in order or at random from the list depending on the
    ``choose_randomly`` parameter.

    If a location requires a shift outside the bounding box of any upstream provider
    the module will skip that location with a warning.

    Args:

        locations (``list`` of locations):

            Locations to center batches around.

        choose_randomly (``bool``):

            Defines whether locations should be picked in order or at random
            from the list.

        extra_data (``list`` of array-like):

            A list of data that will be passed along with the arrays provided
            by this node. This data will be appended as an attribute to the
            dataset so it must be a data format compatible with hdf5.

        jitter (``tuple`` of int):

            How far to allow the point to shift in each direction.
            Default is None, which places the point in the center.
            Chooses uniformly from [loc - jitter, loc + jitter] in each
            direction.
    '''

    def __init__(self, locations, choose_randomly=False, extra_data=None,
                 jitter=None):

        self.coordinates = locations
        self.choose_randomly = choose_randomly
        self.jitter = jitter
        self.loc_i = -1
        self.upstream_spec = None
        self.specified_shift = None

        if extra_data is not None:
            assert len(extra_data) == len(locations),\
                "extra_data (%d) should match the length of specified locations (%d)"%(len(extra_data),\
                len(locations))

        self.extra_data = extra_data

    def setup(self):

        self.upstream_spec = self.get_upstream_provider().spec

        # clear bounding boxes of all provided arrays and points --
        # SpecifiedLocation does know its locations at setup (checks on the fly)
        for key, spec in self.spec.items():
            spec.roi.set_shape(None)
            self.updates(key, spec)

    def prepare(self, request):
        seed(request.random_seed)
        np.random.seed(request.random_seed)
        lcm_voxel_size = self.spec.get_lcm_voxel_size(
            request.array_specs.keys())

        # shift to center
        total_roi = request.get_total_roi()
        request_center = total_roi.get_shape()/2 + total_roi.get_offset()

        self.specified_shift = self._get_next_shift(request_center, lcm_voxel_size)
        while not self.__check_shift(request):
            logger.warning("Location %s (shift %s) skipped"
                           % (self.coordinates[self.loc_i], self.specified_shift))
            self.specified_shift = self._get_next_shift(request_center, lcm_voxel_size)

        # Set shift for all requests
        for specs_type in [request.array_specs, request.graph_specs]:
            for (key, spec) in specs_type.items():
                roi = spec.roi.shift(self.specified_shift)
                specs_type[key].roi = roi

        logger.debug("{}'th ({}) shift selected: {}".format(
            self.loc_i, self.coordinates[self.loc_i], self.specified_shift))

        deps = request
        return deps

    def process(self, batch, request):
        # reset ROIs to request
        for (array_key, spec) in request.array_specs.items():
            batch.arrays[array_key].spec.roi = spec.roi
            if self.extra_data is not None:
                batch.arrays[array_key].attrs['specified_location_extra_data'] =\
                 self.extra_data[self.loc_i]

        for (graph_key, spec) in request.graph_specs.items():
            batch.points[graph_key].spec.roi = spec.roi

        # change shift point locations to lie within roi
        for graph_key in request.graph_specs.keys():
            batch.points[graph_key].shift(-self.specified_shift)

    def _get_next_shift(self, center_shift, voxel_size):
        # gets next coordinate from list

        if self.choose_randomly:
            self.loc_i = randrange(len(self.coordinates))
        else:
            self.loc_i += 1
            if self.loc_i >= len(self.coordinates):
                self.loc_i = 0
                logger.warning('Ran out of specified locations, looping list')
        next_shift = Coordinate(self.coordinates[self.loc_i]) - center_shift

        if self.jitter is not None:
            rnd = []
            for i in range(len(self.jitter)):
                rnd.append(np.random.randint(-self.jitter[i],
                                              self.jitter[i]+1))
            next_shift += Coordinate(rnd)
        logger.debug("Shift before rounding: %s" % str(next_shift))
        # make sure shift is a multiple of voxel size (round to nearest)
        next_shift = Coordinate([int(vs * round(float(shift)/vs)) for vs, shift in zip(voxel_size, next_shift)])
        logger.debug("Shift after rounding: %s" % str(next_shift))
        return next_shift

    def __check_shift(self, request):
        for key, spec in request.items():
            request_roi = spec.roi
            if key in self.upstream_spec:
                provided_roi = self.upstream_spec[key].roi
            else:
                raise Exception(
                    "Requested %s, but upstream does not provide it."%key)
            shifted_roi = request_roi.shift(self.specified_shift)
            if not provided_roi.contains(shifted_roi):
                logger.warning("Provided roi %s for key %s does not contain shifted roi %s"
                             % (provided_roi, key, shifted_roi))
                return False
        return True
