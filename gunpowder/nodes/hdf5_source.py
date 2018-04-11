from gunpowder.ext import h5py
from .hdf5like_source_base import Hdf5LikeSource

class Hdf5Source(Hdf5LikeSource):
    def _open_file(self, filename):
        return h5py.File(filename, 'r')
