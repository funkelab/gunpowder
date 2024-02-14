import logging
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.array import ArrayKey
from gunpowder.roi import Roi
from gunpowder.coordinate import Coordinate
from gunpowder.batch_request import BatchRequest


logger = logging.getLogger(__name__)


class Pad(BatchFilter):
    """Add a constant intensity padding around arrays of another batch
    provider. This is useful if your requested batches can be larger than what
    your source provides.

    Args:

        key (:class:`ArrayKey` or :class:`GraphKey`):

            The array or points set to pad.

        size (:class:`Coordinate` or ``None``):

            The padding to be added. If None, an infinite padding is added. If
            a coordinate, this amount will be added to the ROI in the positive
            and negative direction.

        mode (string):

            One of 'constant' or 'reflect'.
            Default is 'constant'

        value (scalar or ``None``):

            The value to report inside the padding. If not given, 0 is used.
            Only used in case of 'constant' mode.
            Only used for :class:`Array<Arrays>`.
    """

    def __init__(self, key, size, mode="constant", value=None):
        self.key = key
        self.size = size
        self.mode = mode
        if self.mode not in ["constant", "reflect"]:
            raise ValueError(
                "Invalid padding mode %s provided. Must be 'constant' or 'reflect'."
                % self.mode
            )
        self.value = value

    def setup(self):
        self.enable_autoskip()

        assert self.key in self.spec, (
            "Asked to pad %s, but is not provided upstream." % self.key
        )
        assert self.spec[self.key].roi is not None, (
            "Asked to pad %s, but upstream provider doesn't have a ROI for "
            "it." % self.key
        )

        spec = self.spec[self.key].copy()
        if self.size is not None:
            spec.roi = spec.roi.grow(self.size, self.size)
        else:
            spec.roi.shape = Coordinate((None,) * spec.roi.dims)
        self.updates(self.key, spec)

    def prepare(self, request):
        upstream_spec = self.get_upstream_provider().spec

        logger.debug("request: %s" % request)
        logger.debug("upstream spec: %s" % upstream_spec)

        # TODO: remove this?
        if self.key not in request:
            return

        roi = request[self.key].roi.copy()

        # change request to fit into upstream spec
        request[self.key].roi = roi.intersect(upstream_spec[self.key].roi)

        if request[self.key].roi.empty:
            logger.warning(
                "Requested %s ROI %s lies entirely outside of upstream " "ROI %s.",
                self.key,
                roi,
                upstream_spec[self.key].roi,
            )

            # ensure a valid request by asking for empty ROI
            request[self.key].roi = Roi(
                upstream_spec[self.key].roi.offset,
                (0,) * upstream_spec[self.key].roi.dims,
            )

        logger.debug("new request: %s" % request)

        deps = BatchRequest()
        deps[self.key] = request[self.key]
        return deps

    def process(self, batch, request):
        if self.key not in request:
            return

        # restore requested batch size and ROI
        if isinstance(self.key, ArrayKey):
            array = batch.arrays[self.key]
            array.data = self.__expand(
                array.data,
                array.spec.roi / array.spec.voxel_size,
                request[self.key].roi / array.spec.voxel_size,
                self.value if self.value else 0,
            )
            array.spec.roi = request[self.key].roi

        else:
            points = batch.graphs[self.key]
            points.spec.roi = request[self.key].roi

    def __expand(self, a, from_roi, to_roi, value):
        """from_roi and to_roi should be in voxels."""

        logger.debug(
            "expanding array of shape %s from %s to %s", str(a.shape), from_roi, to_roi
        )

        num_channels = len(a.shape) - from_roi.dims
        lower_pad = from_roi.begin - to_roi.begin
        upper_pad = to_roi.end - from_roi.end
        pad_width = [(0, 0)] * num_channels + list(zip(lower_pad, upper_pad))
        if self.mode == "reflect":
            padded = np.pad(a, pad_width, "reflect")
        elif self.mode == "constant":
            padded = np.pad(a, pad_width, "constant", constant_values=value)
        else:
            raise ValueError(
                "Invalid padding mode %s provided. Must be 'constant' or 'reflect'."
                % self.mode
            )
        return padded
