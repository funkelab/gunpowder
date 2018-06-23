import logging
import os

from .batch_filter import BatchFilter
from gunpowder.batch_request import BatchRequest

logger = logging.getLogger(__name__)

class Hdf5LikeWrite(BatchFilter):
    '''Assemble arrays of passing batches in one HDF5-like container. This is
    useful to store chunks produced by :class:`Scan` on disk without keeping
    the larger array in memory. The ROIs of the passing arrays will be used to
    determine the position where to store the data in the dataset.

    Args:

        dataset_names (``dict``, :class:`ArrayKey` -> ``string``):

            A dictionary from array keys to names of the datasets to store them
            in.

        output_dir (``string``):

            The directory to save the container. Will be created, if it does
            not exist.

        output_filename (``string``):

            The output filename of the container. Will be created, if it does
            not exist, otherwise data is overwritten in the existing container.

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

    def _open_file(self, filename):
        raise NotImplementedError('Only implemented in subclasses')

    def __set_voxel_size(self, dataset, voxel_size):
        dataset.attrs['resolution'] = voxel_size

    def __set_offset(self, dataset, offset):
        dataset.attrs['offset'] = offset

    def init_datasets(self, batch):

        try:
            os.makedirs(self.output_dir)
        except:
            pass

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

            filename = os.path.join(self.output_dir, self.output_filename)
            with self._open_file(filename) as data_file:
                dataset = data_file.create_dataset(
                        name=dataset_name,
                        shape=data_shape,
                        compression=self.compression_type,
                        dtype=dtype)

                self.__set_offset(dataset, total_roi.get_offset())
                self.__set_voxel_size(dataset, self.spec[array_key].voxel_size)

    def process(self, batch, request):

        filename = os.path.join(self.output_dir, self.output_filename)

        if not os.path.exists(filename):
            self.init_datasets(batch)

        with self._open_file(filename) as data_file:

            for (array_key, dataset_name) in self.dataset_names.items():

                dataset = data_file[dataset_name]
                roi = batch.arrays[array_key].spec.roi
                data = batch.arrays[array_key].data
                total_roi = self.spec[array_key].roi

                data_roi = (roi - total_roi.get_offset())//self.spec[array_key].voxel_size
                dims = data_roi.dims()
                channel_slices = (slice(None),)*max(0, len(dataset.shape) - dims)
                voxel_slices = data_roi.get_bounding_box()

                dataset[channel_slices + voxel_slices] = batch.arrays[array_key].data


