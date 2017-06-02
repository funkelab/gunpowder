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

    def __init__(
            self,
            filename,
            datasets,
            resolution=None):
        '''Create a new Hdf5Source

        Args

            filename: The HDF5 file.

            datasets: Dictionary of VolumeType -> dataset names that this source offers.

            resolution: tuple, to overwrite the resolution stored in the HDF5 datasets.
        '''

        self.filename = filename
        self.datasets = datasets
        self.specified_resolution = resolution
        self.resolutions = {}

    def setup(self):

        f = h5py.File(self.filename, 'r')

        self.spec = ProviderSpec()
        self.ndims = None
        for (volume_type, ds) in self.datasets.items():

            if ds not in f:
                raise RuntimeError("%s not in %s"%(ds,self.filename))

            dims = f[ds].shape
            self.spec.volumes[volume_type] = Roi((0,)*len(dims), dims)

            if self.ndims is None:
                self.ndims = len(dims)
            else:
                assert self.ndims == len(dims)

            if self.specified_resolution is None:
                if 'resolution' in f[ds].attrs:
                    self.resolutions[volume_type] = tuple(f[ds].attrs['resolution'])
                else:
                    default_resolution = (1,)*self.ndims
                    logger.warning("WARNING: your source does not contain resolution information"
                                   " (no attribute 'resolution' in {} dataset). I will assume {}. "
                                   "This might not be what you want.".format(ds,default_resolution))
                    self.resolutions[volume_type] = default_resolution
            else:
                self.resolutions[volume_type] = self.specified_resolution

        f.close()

    def get_spec(self):
        return self.spec

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        spec = self.get_spec()

        batch = Batch()

        with h5py.File(self.filename, 'r') as f:

            for (volume_type, roi) in request.volumes.items():

                if volume_type not in spec.volumes:
                    raise RuntimeError("Asked for %s which this source does not provide"%volume_type)

                if not spec.volumes[volume_type].contains(roi):
                    raise RuntimeError("%s's ROI %s outside of my ROI %s"%(volume_type,roi,spec.volumes[volume_type]))

                interpolate = {
                    VolumeType.RAW: True,
                    VolumeType.GT_LABELS: False,
                    VolumeType.GT_MASK: False,
                    VolumeType.ALPHA_MASK: True,
                }[volume_type]

                logger.debug("Reading %s in %s..."%(volume_type,roi))
                batch.volumes[volume_type] = Volume(
                        self.__read(f, self.datasets[volume_type], roi),
                        roi=roi,
                        resolution=self.resolutions[volume_type],
                        interpolate=interpolate)

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __read(self, f, ds, roi):

        return np.array(f[ds][roi.get_bounding_box()])

    def __repr__(self):

        return self.filename
