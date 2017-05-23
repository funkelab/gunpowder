import logging

import gunpowder.caffe as caffe
from nodes import *

from batch import Batch
from batch_spec import BatchSpec
from build import build
import batch_provider_tree

from producer_pool import ProducerPool
from coordinate import Coordinate
from roi import Roi

logging.basicConfig(level=logging.INFO)


def set_verbose(verbose=True):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
