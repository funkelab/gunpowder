from __future__ import print_function
import logging
import traceback

import sys
from typing import Optional, Any

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
    dvision = NoSuchModule("dvision")

try:
    import h5py
except ImportError:
    h5py = NoSuchModule("h5py")

try:
    import pyklb
except ImportError:
    pyklb = NoSuchModule("pyklb")

try:
    import tensorflow
except ImportError:
    tensorflow = NoSuchModule("tensorflow")

try:
    import torch
except ImportError:
    torch = NoSuchModule("torch")

try:
    import tensorboardX
except ImportError:
    tensorboardX = NoSuchModule("tensorboardX")

try:
    import malis
except ImportError:
    malis = NoSuchModule("malis")

try:
    import augment
except ImportError:
    augment = NoSuchModule("augment")

ZarrFile: Optional[Any] = None
try:
    import zarr
    from .zarr_file import ZarrFile
except ImportError:
    zarr = NoSuchModule("zarr")
    ZarrFile = None

try:
    import daisy
except ImportError:
    daisy = NoSuchModule("daisy")

try:
    import jax
except ImportError:
    jax = NoSuchModule("jax")

try:
    import jax.numpy as jnp
except ImportError:
    jnp = NoSuchModule("jnp")

try:
    import haiku
except ImportError:
    haiku = NoSuchModule("haiku")

try:
    import optax
except ImportError:
    optax = NoSuchModule("optax")
