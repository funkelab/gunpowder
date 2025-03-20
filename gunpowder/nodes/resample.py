from .batch_filter import BatchFilter
from gunpowder.array import ArrayKey, Array
from gunpowder.batch_request import BatchRequest
from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.roi import Roi
from skimage.transform import rescale
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Resample(BatchFilter):
    """Up- or downsample arrays in a batch to match a given voxel size. Note: Behavior is not a pixel-perfect copy of down/upsample nodes, because this node relies on skimage.transform.rescale to perform non-integer scaling factors.

    Args:

        source (:class:`ArrayKey`):

            The key of the array to resample.

        target_voxel_size (:class:`Coordinate`):

            The voxel size of the target.

        target (:class:`ArrayKey`):

            The key of the array to store the resampled ``source``.

        ndim (``int``, optional):

            Dimensionality of upsampling. This allows users to, for instance, specify against
            resampling in unused z-dimension when processing slices of anisotropic data.
            Default is to use the dimensionality of ``target_voxel_size``.

        interp_order (``int``, optional):

            The order of interpolation. The order has to be in the range 0-5:
                0: Nearest-neighbor
                1: Bi-linear (default)
                2: Bi-quadratic
                3: Bi-cubic
                4: Bi-quartic
                5: Bi-quintic

                Default is 0 if image.dtype is bool or interpolatable is False, and 1 otherwise.

    """

    def __init__(self, source, target_voxel_size, target, ndim=None, interp_order=None):
        assert isinstance(source, ArrayKey)
        assert isinstance(target, ArrayKey)

        self.source = source
        self.target_voxel_size = Coordinate(target_voxel_size)
        self.target = target
        if ndim is None:
            self.ndim = len(target_voxel_size)
        else:
            self.ndim = ndim
        self.interp_order = interp_order

    def setup(self):
        spec = self.spec[self.source].copy()
        source_voxel_size = self.spec[self.source].voxel_size
        spec.voxel_size = self.target_voxel_size
        self.pad = Coordinate(
            (0,) * (len(source_voxel_size) - self.ndim)
            + source_voxel_size[-self.ndim :]
        )

        spec.roi = spec.roi.snap_to_grid(
            np.lcm(source_voxel_size, self.target_voxel_size), mode="shrink"
        )

        self.provides(self.target, spec)
        self.enable_autoskip()

    def prepare(self, request):
        source_voxel_size = self.spec[self.source].voxel_size
        source_request = request[self.target].copy()
        source_request.voxel_size = source_voxel_size
        source_request.roi = source_request.roi.grow(
            self.pad, self.pad
        )  # Pad w/ 1 voxel per side for interpolation to avoid edge effects
        source_request.roi = source_request.roi.snap_to_grid(
            np.lcm(source_voxel_size, self.target_voxel_size), mode="grow"
        )
        source_request.roi = source_request.roi.intersect(
            self.spec[self.source].roi
        ).snap_to_grid(np.lcm(source_voxel_size, self.target_voxel_size), mode="shrink")

        deps = BatchRequest()
        deps[self.source] = source_request

        return deps

    def process(self, batch, request):
        source = batch.arrays[self.source]
        source_data = source.data
        source_voxel_size = self.spec[self.source].voxel_size

        scales = np.array(source_voxel_size) / np.array(self.target_voxel_size)
        scales = (1,) * (source_data.ndim - source_voxel_size.dims) + tuple(scales)

        if self.interp_order != 0 and (
            self.spec[self.source].interpolatable
            or self.spec[self.source].interpolatable is None
        ):
            resampled_data = rescale(
                source_data.astype(np.float32), scales, order=self.interp_order
            ).astype(source_data.dtype)
        else:  # Force nearest-neighbor interpolation for non-interpolatable arrays
            if self.interp_order is not None and self.interp_order != 0:
                logger.warning(
                    "Interpolation other than nearest-neighbor requested for non-interpolatable array. Using nearest-neighbor instead."
                )
            resampled_data = rescale(
                source_data.astype(np.float32), scales, order=0, anti_aliasing=False
            ).astype(source_data.dtype)

        target_spec = source.spec.copy()
        target_spec.roi = Roi(
            source.spec.roi.get_begin(),
            self.target_voxel_size
            * Coordinate(resampled_data.shape[-self.target_voxel_size.dims :]),
        )
        target_spec.voxel_size = self.target_voxel_size
        target_spec.dtype = resampled_data.dtype
        target_array = Array(resampled_data, target_spec)
        target_array.crop(request[self.target].roi)

        # create output array
        outputs = Batch()
        outputs.arrays[self.target] = target_array

        return outputs
