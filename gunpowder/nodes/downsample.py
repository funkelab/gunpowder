from .batch_filter import BatchFilter
from gunpowder.coordinate import Coordinate
from gunpowder.volume import VolumeType, Volume
import logging
import numbers
import numpy as np

logger = logging.getLogger(__name__)

class DownSample(BatchFilter):
    '''Downsample volumes in a batch by given factors.

    Args:

        volume_factors (dict): Dictionary mapping target :class:`VolumeType` to 
            a tuple `(f, volume_type)` of downsampling factor `f` and source 
            :class:`VolumeType`. `f` can be a single integer or a tuple of 
            integers, one for each dimension of the volume to downsample.
    '''

    def __init__(self, volume_factors):

        self.volume_factors = volume_factors

        for output_volume, downsample in volume_factors.items():

            assert isinstance(output_volume, VolumeType)
            assert isinstance(downsample, tuple)
            assert len(downsample) == 2
            f, input_volume = downsample
            assert isinstance(input_volume, VolumeType)
            assert isinstance(f, numbers.Number) or isinstance(f, tuple), "Scaling factor should be a number or a tuple of numbers."

    def setup(self):

        for output_volume, downsample in self.volume_factors.items():
            f, input_volume = downsample
            spec = self.spec[input_volume].copy()
            spec.voxel_size *= f
            self.provides(output_volume, spec)

    def prepare(self, request):

        for output_volume, downsample in self.volume_factors.items():

            f, input_volume = downsample

            if output_volume not in request:
                continue

            logger.debug("preparing downsampling of " + str(input_volume))

            request_roi = request[output_volume].roi
            logger.debug("request ROI is %s"%request_roi)

            # add or merge to batch request
            if input_volume in request:
                request[input_volume].roi = request[input_volume].roi.union(request_roi)
                logger.debug("merging with existing request to %s"%request[input_volume].roi)
            else:
                request[input_volume].roi = request_roi
                logger.debug("adding as new request")

            # remove volume type provided by us
            del request[output_volume]

    def process(self, batch, request):

        for output_volume, downsample in self.volume_factors.items():

            f, input_volume = downsample

            if output_volume not in request:
                continue

            input_roi = batch.volumes[input_volume].spec.roi
            request_roi = request[output_volume].roi

            assert input_roi.contains(request_roi)

            # downsample
            if isinstance(f, tuple):
                slices = tuple(slice(None, None, k) for k in f)
            else:
                slices = tuple(slice(None, None, f) for i in range(input_roi.dims()))

            logger.debug("downsampling %s with %s"%(input_volume, slices))

            crop = batch.volumes[input_volume].crop(request_roi)
            data = crop.data[slices]

            # create output volume
            spec = self.spec[output_volume].copy()
            spec.roi = request_roi
            batch.volumes[output_volume] = Volume(data, spec)

        # restore requested rois
        for output_volume, downsample in self.volume_factors.items():

            f, input_volume = downsample
            if input_volume not in batch.volumes:
                continue

            input_roi = batch.volumes[input_volume].spec.roi
            request_roi = request[input_volume].roi

            if input_roi != request_roi:

                assert input_roi.contains(request_roi)

                logger.debug("restoring original request roi %s of %s from %s"%(request_roi, input_volume, input_roi))
                batch.volumes[input_volume] = batch.volumes[input_volume].crop(request_roi)
