from __future__ import print_function

import logging
import sys
import traceback
from typing import Any, Optional

logger = logging.getLogger(__name__)


class NoSuchModule(object):
    def __init__(self, name):
        self.__name = name
        self.__traceback_str = traceback.format_tb(sys.exc_info()[2])
        errtype, value = sys.exc_info()[:2]
        assert errtype is not None
        self.__exception = errtype(value)

    def __getattr__(self, item):
        raise self.__exception


try:
    import dvision  # ty: ignore[unresolved-import]
except ImportError:
    dvision = NoSuchModule("dvision")

try:
    import h5py
except ImportError:
    h5py = NoSuchModule("h5py")  # ty: ignore[invalid-assignment]

try:
    import pyklb  # ty: ignore[unresolved-import]
except ImportError:
    pyklb = NoSuchModule("pyklb")

try:
    import tensorflow  # ty: ignore[unresolved-import]
except ImportError:
    tensorflow = NoSuchModule("tensorflow")

try:
    import torch  # ty: ignore[unresolved-import]
except ImportError:
    torch = NoSuchModule("torch")

try:
    import tensorboardX  # ty: ignore[unresolved-import]
except ImportError:
    tensorboardX = NoSuchModule("tensorboardX")

try:
    import malis  # ty: ignore[unresolved-import]
except ImportError:
    malis = NoSuchModule("malis")

try:
    import augment
except ImportError:
    augment = NoSuchModule("augment")  # ty: ignore[invalid-assignment]

ZarrFile: Optional[Any] = None
try:
    import zarr

    from .zarr_file import ZarrFile
except ImportError:
    zarr = NoSuchModule("zarr")  # ty: ignore[invalid-assignment]
    ZarrFile = None  # ty: ignore[conflicting-declarations]

try:
    import daisy  # ty: ignore[unresolved-import]
except ImportError:
    daisy = NoSuchModule("daisy")

try:
    import jax  # ty: ignore[unresolved-import]
except ImportError:
    jax = NoSuchModule("jax")

try:
    import jax.numpy as jnp  # ty: ignore[unresolved-import]
except ImportError:
    jnp = NoSuchModule("jnp")

try:
    import haiku  # ty: ignore[unresolved-import]
except ImportError:
    haiku = NoSuchModule("haiku")

try:
    import optax  # ty: ignore[unresolved-import]
except ImportError:
    optax = NoSuchModule("optax")
