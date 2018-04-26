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

        dataset_names (``dict``, :class:`ArrayKey` -> ``string``):

            A dictionary from array keys to names of the datasets to store them
            in.

        output_dir (``string``):

            The directory to save the HDF5 file. Will be created, if it does
            not exist.

        output_filename (``string``):

            The output filename.

        compression_type (``string`` or ``int``):

            Compression strategy.  Legal values are ``gzip``, ``szip``,
            ``lzf``. If an integer between 1 and 10, this indicates ``gzip``
            compression level.

        dataset_dtypes (``dict``, :class:`ArrayKey` -> data type):

            A dictionary from array keys to datatype (eg. ``np.int8``). If
            given, arrays are stored using this type. The original arrays
            within the pipeline remain unchanged.
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

        for (array_key, dataset_name) in self.dataset_names.items():

            logger.debug("Create dataset for %s", array_key)

            assert array_key in self.spec, (
                "Asked to store %s, but is not provided upstream."%array_key)
            assert array_key in batch.arrays, (
                "Asked to store %s, but is not part of batch."%array_key)

            batch_shape = batch.arrays[array_key].data.shape

            total_roi = self.spec[array_key].roi

            assert total_roi is not None, (
                "Provided ROI for %s is not set, I can not guess how large the "
                "HDF5 dataset should be. Make sure that the node that "
                "introduces %s sets its ROI."%(array_key, array_key))

            dims = total_roi.dims()

            # extends of spatial dimensions
            data_shape = total_roi.get_shape()//self.spec[array_key].voxel_size
            logger.debug("Shape in voxels: %s", data_shape)
            # add channel dimensions (if present)
            data_shape = batch_shape[:-dims] + data_shape
            logger.debug("Shape with channel dimensions: %s", data_shape)

            if array_key in self.dataset_dtypes:
                dtype = self.dataset_dtypes[array_key]
            else:
                dtype = batch.arrays[array_key].data.dtype

            dataset = self.file.create_dataset(
                    name=dataset_name,
                    shape=data_shape,
                    compression=self.compression_type,
                    dtype=dtype)

            dataset.attrs['offset'] = total_roi.get_offset()
            dataset.attrs['resolution'] = self.spec[array_key].voxel_size

            self.datasets[array_key] = dataset

    def process(self, batch, request):

        if self.file is None:
            logger.info("Creating HDF file...")
            self.create_output_file(batch)

        for array_key, dataset in self.datasets.items():

            roi = batch.arrays[array_key].spec.roi
            data = batch.arrays[array_key].data
            total_roi = self.spec[array_key].roi

            assert total_roi.contains(roi), (
                "ROI %s of %s not in upstream provided ROI %s"%(
                    roi, array_key, total_roi))

            data_roi = (roi - total_roi.get_offset())//self.spec[array_key].voxel_size
            dims = data_roi.dims()
            channel_slices = (slice(None),)*max(0, len(dataset.shape) - dims)
            voxel_slices = data_roi.get_bounding_box()

            dataset[channel_slices + voxel_slices] = batch.arrays[array_key].data

