from .batch_filter import BatchFilter
from gunpowder.batch_request import BatchRequest
from gunpowder.coordinate import Coordinate
from gunpowder.roi import Roi
import logging
import os

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

        self.dataset_offsets = {}

    def _open_file(self, filename):
        raise NotImplementedError('Only implemented in subclasses')

    def _get_voxel_size(self, dataset):
        return Coordinate(dataset.attrs['resolution'])

    def _get_offset(self, dataset):
        return Coordinate(dataset.attrs['offset'])

    def _set_voxel_size(self, dataset, voxel_size):
        dataset.attrs['resolution'] = voxel_size

    def _set_offset(self, dataset, offset):
        dataset.attrs['offset'] = offset

    def init_datasets(self, batch):

        filename = os.path.join(self.output_dir, self.output_filename)
        logger.debug("Initializing container %s", filename)

        try:
            os.makedirs(self.output_dir)
        except:
            pass

        for (array_key, dataset_name) in self.dataset_names.items():

            logger.debug("Initializing dataset for %s", array_key)

            assert array_key in self.spec, (
                "Asked to store %s, but is not provided upstream."%array_key)
            assert array_key in batch.arrays, (
                "Asked to store %s, but is not part of batch."%array_key)

            array = batch.arrays[array_key]
            dims = array.spec.roi.dims()
            batch_shape = array.data.shape

            with self._open_file(filename) as data_file:

                # if a dataset already exists, read its meta-information (if
                # present)
                if dataset_name in data_file:

                    offset = self._get_offset(data_file[dataset_name]) or Coordinate((0,)*dims)

                else:

                    provided_roi = self.spec[array_key].roi

                    if provided_roi is None:
                        raise RuntimeError(
                            "Dataset %s does not exist in %s, and no ROI is "
                            "provided for %s. I don't know how to initialize "
                            "the dataset."%(dataset_name, filename, array_key))

                    offset = provided_roi.get_offset()
                    voxel_size = array.spec.voxel_size
                    data_shape = provided_roi.get_shape()//voxel_size

                    logger.debug("Shape in voxels: %s", data_shape)
                    # add channel dimensions (if present)
                    data_shape = batch_shape[:-dims] + data_shape
                    logger.debug("Shape with channel dimensions: %s", data_shape)

                    if array_key in self.dataset_dtypes:
                        dtype = self.dataset_dtypes[array_key]
                    else:
                        dtype = batch.arrays[array_key].data.dtype

                    logger.debug(
                        "create_dataset: %s, %s, %s, %s, offset=%s, resolution=%s",
                        dataset_name, data_shape, self.compression_type, dtype,
                        offset, voxel_size)

                    dataset = data_file.create_dataset(
                            name=dataset_name,
                            shape=data_shape,
                            compression=self.compression_type,
                            dtype=dtype)

                    self._set_offset(dataset, offset)
                    self._set_voxel_size(dataset, voxel_size)

                logger.debug(
                    "%s (%s in %s) has offset %s",
                    array_key,
                    dataset_name,
                    filename,
                    offset)
                self.dataset_offsets[array_key] = offset

    def process(self, batch, request):

        filename = os.path.join(self.output_dir, self.output_filename)

        if not self.dataset_offsets:
            self.init_datasets(batch)

        with self._open_file(filename) as data_file:

            for (array_key, dataset_name) in self.dataset_names.items():

                dataset = data_file[dataset_name]

                array_roi = batch.arrays[array_key].spec.roi
                voxel_size = self.spec[array_key].voxel_size
                dims = array_roi.dims()
                channel_slices = (slice(None),)*max(0, len(dataset.shape) - dims)

                dataset_roi = Roi(
                    self.dataset_offsets[array_key],
                    Coordinate(dataset.shape[-dims:])*voxel_size)
                common_roi = array_roi.intersect(dataset_roi)

                if common_roi.empty():
                    logger.warn(
                        "array %s with ROI %s lies outside of dataset ROI %s, "
                        "skipping writing"%(
                            array_key,
                            array_roi,
                            dataset_roi))
                    continue

                dataset_voxel_roi = (common_roi - self.dataset_offsets[array_key])//voxel_size
                dataset_voxel_slices = dataset_voxel_roi.to_slices()
                array_voxel_roi = (common_roi - array_roi.get_offset())//voxel_size
                array_voxel_slices = array_voxel_roi.to_slices()

                logger.debug(
                    "writing %s to voxel coordinates %s"%(
                        array_key,
                        dataset_voxel_roi))

                data = batch.arrays[array_key].data[channel_slices + array_voxel_slices]
                dataset[channel_slices + dataset_voxel_slices] = data
