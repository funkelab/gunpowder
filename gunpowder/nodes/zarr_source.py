from gunpowder.ext import z5py
from .hdf5like_source_base import Hdf5LikeSource


class ZarrSource(Hdf5LikeSource):
    def _open_file(self, filename):
        return z5py.File(filename, use_zarr_format=True)
