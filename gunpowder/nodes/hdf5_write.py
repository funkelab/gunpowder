import logging
import os

from .batch_filter import BatchFilter
from gunpowder.batch_request import BatchRequest
from gunpowder.ext import h5py

logger = logging.getLogger(__name__)

class Hdf5Write(BatchFilter):
    '''Assemble arrays of passing batches in one HDF5 file. This is useful to 
    store chunks produced by :class:`Scan` on disk without keeping the larger 
    array in memory. The ROIs of the passing arrays will be used to determine 
    the position where to store the data in the dataset.

    Args:

        dataset_names (dict): A dictionary from :class:`ArrayKey` to names of 
            the datasets to store them in.

        output_dir (string): The directory to save the HDF5 file. Will be 
            created, if it does not exist.

        output_filename (string): The output filename.

        compression_type (string or int): Compression strategy.  Legal values 
            are 'gzip', 'szip', 'lzf'.  If an integer in range(10), this 
            indicates gzip compression level. Otherwise, an integer indicates 
            the number of a dynamically loaded compression filter. (See 
            h5py.groups.create_dataset())

        dataset_dtypes (dict): A dictionary from :class:`ArrayKey` to datatype
            (eg. np.int8). Array to store is copied and casted to the specified type.
             Original array within the pipeline remains unchanged.
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

        for (array_type, dataset_name) in self.dataset_names.items():

            logger.debug("Create dataset for %s", array_type)

            assert array_type in self.spec, (
                "Asked to store %s, but is not provided upstream."%array_type)
            assert array_type in batch.arrays, (
                "Asked to store %s, but is not part of batch."%array_type)

            batch_shape = batch.arrays[array_type].data.shape

            total_roi = self.spec[array_type].roi
            dims = total_roi.dims()

            # extends of spatial dimensions
            data_shape = total_roi.get_shape()//self.spec[array_type].voxel_size
            logger.debug("Shape in voxels: %s", data_shape)
            # add channel dimensions (if present)
            data_shape = batch_shape[:-dims] + data_shape
            logger.debug("Shape with channel dimensions: %s", data_shape)

            if array_type in self.dataset_dtypes:
                dtype = self.dataset_dtypes[array_type]
            else:
                dtype = batch.arrays[array_type].data.dtype

            dataset = self.file.create_dataset(
                    name=dataset_name,
                    shape=data_shape,
                    compression=self.compression_type,
                    dtype=dtype)

            dataset.attrs['offset'] = total_roi.get_offset()
            dataset.attrs['resolution'] = self.spec[array_type].voxel_size

            self.datasets[array_type] = dataset

    def process(self, batch, request):

        if self.file is None:
            logger.info("Creating HDF file...")
            self.create_output_file(batch)

        for array_type, dataset in self.datasets.items():

            roi = batch.arrays[array_type].spec.roi
            data = batch.arrays[array_type].data
            total_roi = self.spec[array_type].roi

            assert total_roi.contains(roi), (
                "ROI %s of %s not in upstream provided ROI %s"%(
                    roi, array_type, total_roi))

            data_roi = (roi - total_roi.get_offset())//self.spec[array_type].voxel_size
            dims = data_roi.dims()
            channel_slices = (slice(None),)*max(0, len(dataset.shape) - dims)
            voxel_slices = data_roi.get_bounding_box()

            dataset[channel_slices + voxel_slices] = batch.arrays[array_type].data

