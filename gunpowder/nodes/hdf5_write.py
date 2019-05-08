from .hdf5like_write_base import Hdf5LikeWrite
from gunpowder.ext import h5py
import os

class Hdf5Write(Hdf5LikeWrite):
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

    def _open_file(self, filename):
        if os.path.exists(filename):
            return h5py.File(filename, 'r+')
        else:
            return h5py.File(filename, 'w')
