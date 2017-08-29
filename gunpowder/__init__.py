import logging

from .batch import Batch
from .batch_provider_tree import *
from .batch_request import BatchRequest
from .build import build
from .coordinate import Coordinate
from .nodes import *
from .points import PointsTypes, Points, PreSynPoint, PostSynPoint, PointsType, register_points_type, Point
from .points_spec import PointsSpec
from .producer_pool import ProducerPool
from .provider_spec import ProviderSpec
from .roi import Roi
from .volume import register_volume_type, VolumeType, VolumeTypes, Volume
from .volume_spec import VolumeSpec
import gunpowder.caffe
import gunpowder.tensorflow

# logging.basicConfig(level=logging.INFO)

def set_verbose(verbose=True):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
