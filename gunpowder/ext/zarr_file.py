from collections.abc import MutableMapping
from typing import Union

import zarr
from zarr._storage.store import BaseStore


class ZarrFile:
    """To be used as a context manager, similar to h5py.File."""

    def __init__(self, store: Union[BaseStore, MutableMapping, str], mode="a"):
        self.store = store
        self.mode = mode

    def __enter__(self):
        return zarr.open(self.store, mode=self.mode)

    def __exit__(self, *args):
        pass
