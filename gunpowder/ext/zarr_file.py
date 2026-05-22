from typing import Literal

import zarr
from zarr.abc.store import Store

ZarrMode = Literal["r", "r+", "a", "w", "w-"]


class ZarrFile:
    """To be used as a context manager, similar to h5py.File."""

    def __init__(self, store: Store | str, mode: ZarrMode = "a"):
        self.store = store
        self.mode = mode

    def __enter__(self):
        return zarr.open(self.store, mode=self.mode)

    def __exit__(self, *args):
        pass
