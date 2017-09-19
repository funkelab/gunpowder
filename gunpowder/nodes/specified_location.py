import copy
import fractions
import logging
from random import randint

import numpy as np
from skimage.transform import integral_image, integrate
from gunpowder.batch_request import BatchRequest
from gunpowder.coordinate import Coordinate
from gunpowder.roi import Roi
from gunpowder.volume import VolumeTypes
from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class SpecifiedLocation(BatchFilter):
    '''Choses a batch at a random location in the bounding box of the upstream
    provider.

    The random location is chosen such that the batch request roi lies entirely
    inside the provider's roi.

    If `min_masked` (and optionally `mask_volume_type`) are set, only
    batches are returned that have at least the given ratio of masked-in
    voxels. This is in general faster than using the ``Reject`` node, at the
    expense of storing an integral volume of the complete mask.

    If 'focus_points_type' is set, only batches are returned that have at least
    one point of focus_points_type within the roi of PointsTypes.focus_points_type.

    Remark
    ------
    focus_point_type does only work if there are only deterministic nodes upstream

    Args:

        min_masked: If non-zero, require that the random sample contains at
            least that ratio of masked-in voxels.

        mask_volume_type: The volume type to use for mask checks.

        focus_points_type: gunpowder.PointsTypes, PointsTypes considered when
            looking for good location of batch s.t. at least one point of this
            PointsTypes is contained in batch
    '''

    def __init__(self, specified_locations, choose_randomly = False):

        self.locs = specified_locations
        self.choose_randomly = choose_randomly
        self.loc_i = 0


    def setup(self):

        self.upstream_spec = self.get_upstream_provider().spec
        self.upstream_roi = self.upstream_spec.get_total_roi()

        if self.upstream_roi is None:
            raise RuntimeError("Can not draw random samples from a provider that does not have a bounding box.")  

        # clear bounding boxes of all provided volumes and points -- 
        # RandomLocation does not have limits (offsets are ignored)
        for identifier, spec in self.spec.items():
            spec.roi = None
            self.updates(identifier, spec)

    def prepare(self, request):

        shift_roi = None

        for identifier, spec in request.items():
            request_roi = spec.roi
            if identifier in self.upstream_spec:
                provided_roi = self.upstream_spec[identifier].roi
            else:
                raise Exception(
                    "Requested %s, but upstream does not provide "
                    "it."%identifier)
            type_shift_roi = provided_roi.shift(-request_roi.get_begin()).grow((0,0,0),-request_roi.get_shape())

            if shift_roi is None:
                shift_roi = type_shift_roi
            else:
                shift_roi = shift_roi.intersect(type_shift_roi)

        logger.debug("valid shifts for request in " + str(shift_roi))

        # shift to center
        center_shift = np.asarray(spec.roi.get_shape())/2

        # shift request ROIs      
        self.specified_shift = self._get_next_shift(center_shift)
      
        # Make sure shift fits in roi of all request types
        for specs_type in [request.volume_specs, request.points_specs]:
            for (type, spec) in specs_type.items():

                roi = spec.roi.shift(self.specified_shift)

                while not self.upstream_spec[type].roi.contains(roi):
                    logger.warning("selected roi {} doesn't fit in upstream provider.\n Skipping this location...".format(roi) )
               
                    self.specified_shift = self._get_next_shift(center_shift) 
                    roi = spec.roi.shift(self.specified_shift)      
                             
        # Once an acceptable shift has been found, set that for all requests
        for specs_type in [request.volume_specs, request.points_specs]:
            for (type, spec) in specs_type.items():
                roi = spec.roi.shift(self.specified_shift)
                specs_type[type].roi = roi

        logger.debug("{}'th shift selected: {}".format(self.loc_i, self.specified_shift) )

    def process(self, batch, request):
        # reset ROIs to request
        for (volume_type, spec) in request.volume_specs.items():
            batch.volumes[volume_type].spec.roi = spec.roi
        for (points_type, spec) in request.points_specs.items():
            batch.points[points_type].spec.roi = spec.roi

        # change shift point locations to lie within roi
        for points_type in request.points_specs.keys():
            for point_id, point in batch.points[points_type].data.items():
                batch.points[points_type].data[point_id].location -= self.specified_shift

    # get next shift from list
    def _get_next_shift(self, center_shift):
        if self.choose_randomly:
            next_shift = Coordinate(random.choice(self.locs) - center_shift)
        else:
            next_shift = Coordinate(self.locs[self.loc_i] - center_shift)
            self.loc_i += 1
            if self.loc_i >= len(self.locs):
               self.loc_i = 0
        return next_shift
