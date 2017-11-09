import logging
import os

from .batch_filter import BatchFilter
from gunpowder.batch_request import BatchRequest
from gunpowder.ext import h5py
from gunpowder.volume import VolumeTypes

logger = logging.getLogger(__name__)

class Hdf5Write(BatchFilter):
    '''Assemble volumes of passing batches in one HDF5 file. This is useful to 
    store chunks produced by :class:`Scan` on disk without keeping the larger 
    volume in memory. The ROIs of the passing volumes will be used to determine 
    the position where to store the data in the dataset.

    Args:

        dataset_names (dict): A dictionary from :class:`VolumeType` to names of 
            the datasets to store them in.

        output_dir (string): The directory to save the HDF5 file. Will be 
            created, if it does not exist.

        output_filename (string): The output filename.

        compression_type (string or int): Compression strategy.  Legal values 
            are 'gzip', 'szip', 'lzf'.  If an integer in range(10), this 
            indicates gzip compression level. Otherwise, an integer indicates 
            the number of a dynamically loaded compression filter. (See 
            h5py.groups.create_dataset())

        dataset_dtypes (dict): A dictionary from :class:`VolumeType` to datatype
            (eg. np.int8). Volume to store is copied and casted to the specified type.
             Original volume within the pipeline remains unchanged.
        '''

    def __init__(
            self,
            dataset_names,
            output_dir='.',
            output_filename='output.hdf',
            compression_type=None,
            dataset_dtypes=None):

        self.dataset_names = dataset_names
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.compression_type = compression_type
        if dataset_dtypes is None:
            self.dataset_dtypes = {}
        else:
            self.dataset_dtypes = dataset_dtypes
        self.file = None

    def create_output_file(self, batch):

        try:
            os.makedirs(self.output_dir)
        except:
            pass

        self.file = h5py.File(os.path.join(self.output_dir, self.output_filename), 'w')
        self.datasets = {}

        for (volume_type, dataset_name) in self.dataset_names.items():

            assert volume_type in self.spec, "Asked to store %s, but is not provided upstream."%volume_type

            total_roi = self.spec[volume_type].roi
            data_shape = total_roi.get_shape()//self.spec[volume_type].voxel_size

            if volume_type in self.dataset_dtypes:
                dtype = self.dataset_dtypes[volume_type]
            else:
                dtype = batch.volumes[volume_type].data.dtype

            dataset = self.file.create_dataset(
                    name=dataset_name,
                    shape=data_shape,
                    compression=self.compression_type,
                    dtype=dtype)

            dataset.attrs['offset'] = total_roi.get_offset()
            dataset.attrs['resolution'] = self.spec[volume_type].voxel_size

            self.datasets[volume_type] = dataset

    def process(self, batch, request):

        if self.file is None:
            logger.info("Creating HDF file...")
            self.create_output_file(batch)

        for volume_type, dataset in self.datasets.items():

            roi = batch.volumes[volume_type].spec.roi
            data = batch.volumes[volume_type].data
            total_roi = self.spec[volume_type].roi

            assert total_roi.contains(roi), "ROI %s of %s not in upstream provided ROI %s"%(roi, volume_type, total_roi)
            data_roi = (roi - total_roi.get_offset())//self.spec[volume_type].voxel_size
            dataset[data_roi.get_bounding_box()] = batch.volumes[volume_type].data
