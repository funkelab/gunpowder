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
    '''Up- or downsample arrays in a batch to match a given voxel size.

    Args:

        source (:class:`ArrayKey`):

            The key of the array to resample.

        target_voxel_size (:class:`Coordinate`):

            The voxel size of the target.

        target (:class:`ArrayKey`):

            The key of the array to store the resampled ``source``.
    '''

    def __init__(self, source, target_voxel_size, target, interp_order=None):

        assert isinstance(source, ArrayKey)
        assert isinstance(target, ArrayKey)

        self.source = source
        self.target_voxel_size = Coordinate(target_voxel_size)
        self.target = target
        self.interp_order = interp_order

    def setup(self):

        spec = self.spec[self.source].copy()
        spec.voxel_size = self.target_voxel_size
        # spec.roi = spec.roi.snap_to_grid(spec.voxel_size, mode='shrink')
        self.provides(self.target, spec)
        self.enable_autoskip()

    def prepare(self, request):

        # source_voxel_size = 2
        # |-----------------|   (source array ROI)
        # * * * * * * * * * x
        #
        # target_voxel_size = 1
        # |-----------------|   (source array ROI)
        # ******************* (odd)

        source_voxel_size = self.spec[self.source].voxel_size

        source_request = request[self.target].copy()
        source_request.voxel_size = source_voxel_size
        source_request.roi = source_request.roi.snap_to_grid(
            source_voxel_size,
            mode='grow')

        deps = BatchRequest()
        deps[self.source] = source_request

        return deps

    def process(self, batch, request):
        source = batch.arrays[self.source]
        source_data = source.data
        source_voxel_size = self.spec[self.source].voxel_size

        scales = np.array(source_voxel_size) / np.array(self.target_voxel_size)

        if self.spec[self.source].interpolatable:
            resampled_data = rescale(source_data, scales, order=self.interp_order)
            resampled_data = resampled_data.astype(np.float32)
        else: # Force nearest-neighbor interpolation for non-interpolatable arrays
            if self.interp_order is not None and self.interp_order != 0:
                logger.warning('Interpolation other than nearest-neighbor requested for non-interpolatable array. Using nearest-neighbor instead.')
            resampled_data = rescale(source_data, scales, order=0, anti_aliasing=False)

        target_spec = source.spec.copy()
        target_spec.roi = Roi(
            source.spec.roi.get_begin(),
            self.target_voxel_size * resampled_data.shape
        )
        target_spec.voxel_size = self.target_voxel_size
        target_spec.dtype = resampled_data.dtype
        target_array = Array(resampled_data, target_spec)
        target_array.crop(request[self.target].roi)

        # create output array
        outputs = Batch()
        outputs.arrays[self.target] = target_array

        return outputs