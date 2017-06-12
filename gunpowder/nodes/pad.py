import copy
import logging
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.roi import Roi
from gunpowder.coordinate import Coordinate
from gunpowder.volume import VolumeType

logger = logging.getLogger(__name__)

class Pad(BatchFilter):
    '''Add a constant intensity padding around volumes of another batch 
    provider. This is useful if your requested batches can be larger than what 
    your source provides.
    '''

    def __init__(self, pad_sizes, pad_values=None):
        '''
        Args:

            pad_sizes: dict, VolumeType -> [None,Coordinate]

                Specifies the padding to be added to each volume type. If None, 
                an infinite padding is added. If a Coordinate, this amount will 
                be added to the ROI in the positive and negative direction.

            pad_values: dict, VolumeType -> value or None

                The values to report inside the padding. If not given, 0 is 
                used.
        '''

        self.pad_sizes = pad_sizes
        if pad_values is None:
            self.pad_values = {}
        else:
            self.pad_values = pad_values

    def setup(self):

        self.upstream_spec = self.get_upstream_provider().get_spec()
        self.spec = copy.deepcopy(self.upstream_spec)

        for (volume_type, pad_size) in self.pad_sizes.items():

            assert volume_type in self.spec.volumes, "Asked to pad %s, but is not provided upstream."%volume_type
            assert self.spec.volumes[volume_type] is not None, "Asked to pad %s, but upstream provider doesn't have a ROI for it."%volume_type

            if pad_size is not None:
                self.spec.volumes[volume_type] = self.upstream_spec.volumes[volume_type].grow(pad_size, pad_size)

        logger.debug("upstream spec: " + str(self.upstream_spec))
        logger.debug("provided spec:" + str(self.spec))

    def get_spec(self):
        return self.spec

    def prepare(self, request):

        logger.debug("request: %s"%request)
        logger.debug("upstream spec: %s"%self.upstream_spec)

        # remember request
        self.request = copy.deepcopy(request)

        for volume_type in self.pad_sizes.keys():

            if volume_type not in request.volumes:
                continue
            roi = request.volumes[volume_type]

            # check out-of-bounds
            # TODO: this should be moved to super class, this should hold for any 
            # batch provider
            if self.pad_sizes[volume_type] is not None:
                if not self.spec.volumes[volume_type].intersects(roi):
                    raise RuntimeError("%s ROI %s lies outside of padded ROI %s"%(volume_type,roi,self.spec.volumes[volume_type]))

            # change request to fit into upstream spec
            request.volumes[volume_type] = roi.intersect(self.upstream_spec.volumes[volume_type])

            if request.volumes[volume_type] is None:

                logger.warning("Requested %s ROI lies entirely outside of upstream ROI."%volume_type)

                # ensure a valid request by asking for empty ROI
                request.volumes[volume_type] = Roi(
                        self.upstream_spec.volumes[volume_type].get_offset(),
                        (0,)*self.upstream_spec.volumes[volume_type].dims()
                )

        logger.debug("new request: %s"%request)

    def process(self, batch, request):

        # restore requested batch size and ROI
        for (volume_type, volume) in batch.volumes.items():

            volume.data = self.__expand(
                    volume.data,
                    volume.roi,
                    self.request.volumes[volume_type],
                    self.pad_values[volume_type] if volume_type in self.pad_values else 0
            )
            volume.roi = self.request.volumes[volume_type]

    def __expand(self, a, from_roi, to_roi, value):

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
