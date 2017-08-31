import copy
import logging
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.roi import Roi
from gunpowder.coordinate import Coordinate
from gunpowder.volume import VolumeTypes

logger = logging.getLogger(__name__)

class Pad(BatchFilter):
    '''Add a constant intensity padding around volumes of another batch 
    provider. This is useful if your requested batches can be larger than what 
    your source provides.

    Args:

        pad_sizes(dict, VolumeTypes -> [None,Coordinate]): Specifies the padding 
            to be added to each volume type. If None, an infinite padding is 
            added. If a Coordinate, this amount will be added to the ROI in the 
            positive and negative direction.

        pad_values(dict, VolumeTypes -> value or None): The values to report 
            inside the padding. If not given, 0 is used.
    '''

    def __init__(self, pad_sizes, pad_values=None):

        self.pad_sizes = pad_sizes
        if pad_values is None:
            self.pad_values = {}
        else:
            self.pad_values = pad_values

    def setup(self):

        for (volume_type, pad_size) in self.pad_sizes.items():

            assert volume_type in self.spec, "Asked to pad %s, but is not provided upstream."%volume_type
            assert self.spec[volume_type].roi is not None, "Asked to pad %s, but upstream provider doesn't have a ROI for it."%volume_type

            spec = self.spec[volume_type].copy()
            if pad_size is not None:
                spec.roi = spec.roi.grow(pad_size, pad_size)
            else:
                spec.roi = None
            self.updates(volume_type, spec)

    def prepare(self, request):

        upstream_spec = self.get_upstream_provider().spec

        logger.debug("request: %s"%request)
        logger.debug("upstream spec: %s"%upstream_spec)

        for volume_type in self.pad_sizes.keys():

            if volume_type not in request:
                continue
            roi = request[volume_type].roi

            # change request to fit into upstream spec
            request[volume_type].roi = roi.intersect(upstream_spec[volume_type].roi)

            if request[volume_type].roi is None:

                logger.warning("Requested %s ROI lies entirely outside of upstream ROI."%volume_type)

                # ensure a valid request by asking for empty ROI
                request[volume_type].roi = Roi(
                        upstream_spec[volume_type].roi.get_offset(),
                        (0,)*upstream_spec[volume_type].roi.dims()
                )

        logger.debug("new request: %s"%request)

    def process(self, batch, request):

        # restore requested batch size and ROI
        for (volume_type, volume) in batch.volumes.items():

            volume.data = self.__expand(
                    volume.data,
                    volume.spec.roi/volume.spec.voxel_size,
                    request[volume_type].roi/volume.spec.voxel_size,
                    self.pad_values[volume_type] if volume_type in self.pad_values else 0
            )
            volume.spec.roi = request[volume_type].roi

    def __expand(self, a, from_roi, to_roi, value):
        '''from_roi and to_roi should be in voxels.'''

        logger.debug("expanding array of shape %s from %s to %s"%(str(a.shape), from_roi, to_roi))

        b = np.zeros(to_roi.get_shape(), dtype=a.dtype)
        if value != 0:
            b[:] = value

        shift = tuple(-x for x in to_roi.get_offset())
        logger.debug("shifting 'from' by " + str(shift))
        a_in_b = from_roi.shift(shift).get_bounding_box()

        logger.debug("target shape is " + str(b.shape))
        logger.debug("target slice is " + str(a_in_b))

        b[a_in_b] = a

        return b
