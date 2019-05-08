import zarr

class ZarrFile():
    '''To be used as a context manager, similar to h5py.File.'''

    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        return zarr.open(self.filename, mode=self.mode)

    def __exit__(self, *args):
        pass
