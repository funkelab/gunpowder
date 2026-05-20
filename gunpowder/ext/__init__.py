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
        self.__exception = errtype(value)

    def __getattr__(self, item):
        raise self.__exception


try:
    import dvision
except ImportError:
    dvision = NoSuchModule("dvision")  # type: ignore

try:
    import h5py
except ImportError:
    h5py = NoSuchModule("h5py")  # type: ignore

try:
    import pyklb
except ImportError:
    pyklb = NoSuchModule("pyklb")  # type: ignore

try:
    import tensorflow
except ImportError:
    tensorflow = NoSuchModule("tensorflow")  # type: ignore

try:
    import torch
except ImportError:
    torch = NoSuchModule("torch")  # type: ignore

try:
    import tensorboardX
except ImportError:
    tensorboardX = NoSuchModule("tensorboardX")  # type: ignore

try:
    import malis
except ImportError:
    malis = NoSuchModule("malis")  # type: ignore

try:
    import augment
except ImportError:
    augment = NoSuchModule("augment")  # type: ignore

ZarrFile: Optional[Any] = None
try:
    import zarr

    from .zarr_file import ZarrFile
except ImportError:
    zarr = NoSuchModule("zarr")  # type: ignore
    ZarrFile = None

try:
    import daisy
except ImportError:
    daisy = NoSuchModule("daisy")  # type: ignore

try:
    import jax
except ImportError:
    jax = NoSuchModule("jax")  # type: ignore

try:
    import jax.numpy as jnp
except ImportError:
    jnp = NoSuchModule("jnp")  # type: ignore

try:
    import haiku
except ImportError:
    haiku = NoSuchModule("haiku")  # type: ignore

try:
    import optax
except ImportError:
    optax = NoSuchModule("optax")  # type: ignore
