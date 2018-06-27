from .hdf5like_write_base import Hdf5LikeWrite
from gunpowder.coordinate import Coordinate
from gunpowder.ext import z5py
import logging
import os
import traceback

logger = logging.getLogger(__name__)

class N5Write(Hdf5LikeWrite):
    '''Assemble arrays of passing batches in one N5 container. This is useful
    to store chunks produced by :class:`Scan` on disk without keeping the
    larger array in memory. The ROIs of the passing arrays will be used to
    determine the position where to store the data in the dataset.

    Args:

        dataset_names (``dict``, :class:`ArrayKey` -> ``string``):

            A dictionary from array keys to names of the datasets to store them
            in.

        output_dir (``string``):

            The directory to save the N5 container. Will be created, if it does
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

    def _get_voxel_size(self, dataset):

        logger.debug('Voxel size being reversed to account for N5 using column-major ordering')
        try:
            return Coordinate(dataset.attrs['resolution'][::-1])
        except Exception:
            logger.error(traceback.format_exc())
            return None

    def _get_offset(self, dataset):

        logger.debug('Offset being reversed to account for N5 using column-major ordering')
        try:
            return Coordinate(dataset.attrs['offset'][::-1])
        except Exception:
            logger.error(traceback.format_exc())
            return None

    def _set_voxel_size(self, dataset, voxel_size):

        logger.debug('Voxel size being reversed to account for N5 using column-major ordering')
        dataset.attrs['resolution'] = voxel_size[::-1]

    def _set_offset(self, dataset, offset):

        logger.debug('Offset being reversed to account for N5 using column-major ordering')
        dataset.attrs['offset'] = offset[::-1]

    def _open_file(self, filename):
        return z5py.File(filename, use_zarr_format=False)

