import zarr
from zarr.storage import Store


class ZarrFile():
    '''To be used as a context manager, similar to h5py.File.'''

    def __init__(self, filename, store: Store = None, mode='a'):
        self.filename = filename
        self.store = store
        self.mode = mode

    def __enter__(self):
        if self.store is None:
            return zarr.open(self.filename, mode=self.mode)
        else:
            return zarr.open(self.store, mode=self.mode)

    def __exit__(self, *args):
        pass
