from __future__ import absolute_import

from . import nodes
from .nodes import *

from .batch import Batch
from .batch_provider_tree import *
from .batch_request import BatchRequest
from .build import build
from .coordinate import Coordinate
from .points import PointsTypes, Points, PreSynPoint, PostSynPoint, PointsType, register_points_type, Point
from .points_spec import PointsSpec
from .producer_pool import ProducerPool
from .provider_spec import ProviderSpec
from .roi import Roi
from .volume import register_volume_type, ArrayType, ArrayTypes, Array
from .volume_spec import ArraySpec
import gunpowder.caffe
import gunpowder.tensorflow
import gunpowder.contrib

import logging

# logging.basicConfig(level=logging.INFO)

def set_verbose(verbose=True):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
