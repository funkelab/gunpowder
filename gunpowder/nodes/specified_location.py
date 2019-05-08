from random import choice
import logging
import numpy as np

from gunpowder.coordinate import Coordinate
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
    '''

    def __init__(self, locations, choose_randomly=False, extra_data=None):

        self.coordinates = locations
        self.choose_randomly = choose_randomly
        self.loc_i = 0
        self.upstream_spec = None
        self.upstream_roi = None
        self.specified_shift = None

        if extra_data is not None:
            assert len(extra_data) == len(locations),\
                "extra_data (%d) should match the length of specified locations (%d)"%(len(extra_data),\
                len(locations))

        self.extra_data = extra_data

    def setup(self):

        self.upstream_spec = self.get_upstream_provider().spec
        self.upstream_roi = self.upstream_spec.get_total_roi()

        if self.upstream_roi is None:
            raise RuntimeError("Can not draw random samples from a provider\
                that does not have a bounding box.")

        # clear bounding boxes of all provided arrays and points --
        # SpecifiedLocation does know its locations at setup (checks on the fly)
        for key, spec in self.spec.items():
            spec.roi.set_shape(None)
            self.updates(key, spec)

    def prepare(self, request):

        shift_roi = None

        for key, spec in request.items():
            request_roi = spec.roi
            if key in self.upstream_spec:
                provided_roi = self.upstream_spec[key].roi
            else:
                raise Exception(
                    "Requested %s, but upstream does not provide it."%key)
            key_shift_roi = provided_roi.shift(-request_roi.get_begin()).grow((0, 0, 0),
                                                    -request_roi.get_shape())

            if shift_roi is None:
                shift_roi = key_shift_roi
            else:
                shift_roi = shift_roi.intersect(key_shift_roi)

        logger.debug("valid shifts for request in " + str(shift_roi))

        # shift to center
        center_shift = spec.roi.get_shape()/2 + spec.roi.get_offset()

        self.specified_shift = self._get_next_shift(center_shift)

        # Set shift for all requests
        for specs_type in [request.array_specs, request.points_specs]:
            for (key, spec) in specs_type.items():
                roi = spec.roi.shift(self.specified_shift)
                specs_type[key].roi = roi

        logger.debug("{}'th shift selected: {}".format(self.loc_i, self.specified_shift))

    def process(self, batch, request):
        # reset ROIs to request
        for (array_key, spec) in request.array_specs.items():
            batch.arrays[array_key].spec.roi = spec.roi
            if self.extra_data is not None:
                batch.arrays[array_key].attrs['specified_location_extra_data'] =\
                 self.extra_data[self.loc_i]

        for (points_key, spec) in request.points_specs.items():
            batch.points[points_key].spec.roi = spec.roi

        # change shift point locations to lie within roi
        for points_key in request.points_specs.keys():
            for point_id, point in batch.points[points_key].data.items():
                batch.points[points_key].data[point_id].location -= self.specified_shift

    def _get_next_shift(self, center_shift):
        # gets next corrdinate from list

        if self.choose_randomly:
            next_shift = choice(self.coordinates) - center_shift
        else:
            next_shift = Coordinate(self.coordinates[self.loc_i] - center_shift)
            self.loc_i += 1
            if self.loc_i >= len(self.coordinates):
                self.loc_i = 0
                logger.warning('Ran out of specified locations, looping list')
        return next_shift
