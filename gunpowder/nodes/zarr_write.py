from .hdf5like_write_base import Hdf5LikeWrite
from gunpowder.coordinate import Coordinate
from gunpowder.ext import ZarrFile
from gunpowder.compat import ensure_str
import logging
import os
import traceback

logger = logging.getLogger(__name__)

class ZarrWrite(Hdf5LikeWrite):
    '''Assemble arrays of passing batches in one zarr container. This is useful
    to store chunks produced by :class:`Scan` on disk without keeping the
    larger array in memory. The ROIs of the passing arrays will be used to
    determine the position where to store the data in the dataset.

    Args:

        dataset_names (``dict``, :class:`ArrayKey` -> ``string``):

            A dictionary from array keys to names of the datasets to store them
            in.

        output_dir (``string``):

            The directory to save the zarr container. Will be created, if it does
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

        if self.output_filename.endswith('.n5'):
            return Coordinate(dataset.attrs['resolution'][::-1])
        else:
            return Coordinate(dataset.attrs['resolution'])

    def _get_offset(self, dataset):

        if self.output_filename.endswith('.n5'):
            return Coordinate(dataset.attrs['offset'][::-1])
        else:
            return Coordinate(dataset.attrs['offset'])

    def _set_voxel_size(self, dataset, voxel_size):

        if self.output_filename.endswith('.n5'):
            dataset.attrs['resolution'] = voxel_size[::-1]
        else:
            dataset.attrs['resolution'] = voxel_size

    def _set_offset(self, dataset, offset):

        if self.output_filename.endswith('.n5'):
            dataset.attrs['offset'] = offset[::-1]
        else:
            dataset.attrs['offset'] = offset

    def _open_file(self, filename):
        return ZarrFile(ensure_str(filename), mode='a')
