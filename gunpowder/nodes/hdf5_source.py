import logging
import numpy as np

from .batch_provider import BatchProvider
from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.ext import h5py
from gunpowder.profiling import Timing
from gunpowder.provider_spec import ProviderSpec
from gunpowder.roi import Roi
from gunpowder.volume import Volume, VolumeType

logger = logging.getLogger(__name__)

class Hdf5Source(BatchProvider):

    def __init__(self, filename, raw_dataset, gt_dataset=None, gt_mask_dataset=None, resolution=None):

        self.filename = filename
        self.raw_dataset = raw_dataset
        self.gt_dataset = gt_dataset
        self.gt_mask_dataset = gt_mask_dataset
        self.specified_resolution = resolution

    def setup(self):

        f = h5py.File(self.filename, 'r')

        self.spec = ProviderSpec()
        self.ndims = None
        for volume_type in [VolumeType.RAW, VolumeType.GT_LABELS, VolumeType.GT_MASK]:

            ds = {
                    VolumeType.RAW: self.raw_dataset,
                    VolumeType.GT_LABELS: self.gt_dataset,
                    VolumeType.GT_MASK: self.gt_mask_dataset,
            }[volume_type]

            if ds is None:
                continue

            if ds not in f:
                raise RuntimeError("%s not in %s"%(ds,self.filename))

            dims = f[ds].shape
            self.spec.volumes[volume_type] = Roi((0,)*len(dims), dims)

            if self.ndims is None:
                self.ndims = len(dims)
            else:
                assert self.ndims == len(dims)

        f.close()

    def get_spec(self):
        return self.spec

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        spec = self.get_spec()

        batch = Batch()
        logger.debug("providing batch with resolution of {}".format(self.resolution))

        with h5py.File(self.filename, 'r') as f:

            for (volume_type, roi) in request.volumes.items():

                if volume_type not in spec.volumes:
                    raise RuntimeError("Asked for %s which this source does not provide"%volume_type)

                if not spec.volumes[volume_type].contains(roi):
                    raise RuntimeError("%s's ROI %s outside of my ROI %s"%(volume_type,roi,spec.volumes[volume_type]))

                dataset, interpolate = {
                    VolumeType.RAW: (self.raw_dataset, True),
                    VolumeType.GT_LABELS: (self.gt_dataset, False),
                    VolumeType.GT_MASK: (self.gt_mask_dataset, False),
                }[volume_type]

                logger.debug("Reading %s in %s..."%(volume_type,roi))
                batch.volumes[volume_type] = Volume(
                        self.__read(f, dataset, roi),
                        roi=roi,
                        resolution=self.resolution,
                        interpolate=interpolate)

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __read(self, f, ds, roi):

        return np.array(f[ds][roi.get_bounding_box()])

    def __repr__(self):

        return self.filename

    @property
    def resolution(self):
        if self.specified_resolution is not None:
            return self.specified_resolution
        else:
            try:
                with h5py.File(self.filename, 'r') as f:
                    return tuple(f[self.raw_dataset].attrs['resolution'])
            except KeyError:
                default_resolution = (1,) * self.ndims
                logger.warning("WARNING: your source does not contain resolution information"
                               " (no attribute 'resolution' in raw dataset). I will assume {}. "
                               "This might not be what you want.".format(default_resolution))
                return default_resolution
