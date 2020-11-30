from __future__ import print_function
import logging
import traceback

import sys

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
except ImportError as e:
    dvision = NoSuchModule('dvision')

try:
    import h5py
except ImportError as e:
    h5py = NoSuchModule('h5py')

try:
    import pyklb
except ImportError as e:
    pyklb = NoSuchModule('pyklb')

try:
    import tensorflow
except ImportError as e:
    tensorflow = NoSuchModule('tensorflow')

try:
    import torch
except ImportError as e:
    torch = NoSuchModule('torch')

try:
    import tensorboardX
except ImportError as e:
    tensorboardX = NoSuchModule('tensorboardX')

try:
    import malis
except ImportError as e:
    malis = NoSuchModule('malis')

try:
    import augment
except ImportError as e:
    augment = NoSuchModule('augment')

try:
    import zarr
    from .zarr_file import ZarrFile
except ImportError as e:
    zarr = NoSuchModule('zarr')
    ZarrFile = None

try:
    import daisy
except ImportError as e:
    daisy = NoSuchModule('daisy')
