import logging
import numpy as np

from .batch_provider import BatchProvider
from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.ext import h5py
from gunpowder.profiling import Timing
from gunpowder.provider_spec import ProviderSpec
from gunpowder.roi import Roi
from gunpowder.volume import Volume, VolumeTypes

logger = logging.getLogger(__name__)

class Hdf5Source(BatchProvider):
    '''An HDF5 data source.

    Provides volumes from HDF5 datasets for each volume type given. If the 
    attribute ``resolution`` is set in an HDF5 dataset, it will be used for the 
    resolution of the volume. If the attribute ``offset`` is set in an HDF5 
    dataset, it will be used as the offset of the :class:`Roi` provided by this 
    node. It is assumed that the offset is given in world units. Since 
    ``gunpowder`` ROIs are in voxels, the ``offset`` attribute will be divided 
    by the ``resolution``.

    Args:

        filename (string): The HDF5 file.

        datasets (dict): Dictionary of VolumeType -> dataset names that this source offers.

        resolution (tuple): Overwrite the resolution stored in the HDF5 datasets.
    '''

    def __init__(
            self,
            filename,
            datasets,
            resolution=None):

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

            if self.ndims is None:
                self.ndims = len(dims)
            else:
                assert self.ndims == len(dims)

            if self.specified_resolution is None:
                if 'resolution' in f[ds].attrs:
                    self.resolutions[volume_type] = Coordinate(f[ds].attrs['resolution'])
                else:
                    default_resolution = Coordinate((1,)*self.ndims)
                    logger.warning("WARNING: your source does not contain resolution information"
                                   " (no attribute 'resolution' in {} dataset). I will assume {}. "
                                   "This might not be what you want.".format(ds,default_resolution))
                    self.resolutions[volume_type] = default_resolution
            else:
                self.resolutions[volume_type] = self.specified_resolution

            if 'offset' in f[ds].attrs:
                offset = Coordinate(f[ds].attrs['offset'])
                offset /= self.resolutions[volume_type]
            else:
                offset = Coordinate((0,)*self.ndims)

            self.spec.volumes[volume_type] = Roi(offset, dims)

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

                logger.debug("Reading %s in %s..."%(volume_type,roi))

                # shift request roi into dataset
                dataset_roi = roi.shift(-spec.volumes[volume_type].get_offset())

                batch.volumes[volume_type] = Volume(
                        self.__read(f, self.datasets[volume_type], dataset_roi),
                        roi=roi,
                        resolution=self.resolutions[volume_type])

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __read(self, f, ds, roi):

        return np.array(f[ds][roi.get_bounding_box()])

    def __repr__(self):

        return self.filename
