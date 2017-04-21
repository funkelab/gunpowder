import logging
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
        "-v", "--verbose",
        action="store_true")
args = parser.parse_args()
if args.verbose:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

from batch_spec import BatchSpec
from batch import Batch
from batch_filter import BatchFilter
from hdf5_source import Hdf5Source
from random_location import RandomLocation
from random_provider import RandomProvider
from elastic_augmentation import ElasticAugmentation
from snapshot import Snapshot
from precache import PreCache
from grow_boundary import GrowBoundary
from reject import Reject
from simple_augment import SimpleAugment
from defect_augment import DefectAugment
from exclude_labels import ExcludeLabels
from add_gt_affinities import AddGtAffinities
from intensity_augment import IntensityAugment
from normalize import Normalize
from intensity_scale_shift import IntensityScaleShift
from zero_out_const_sections import ZeroOutConstSections
from padding import Padding
from print_profiling_stats import PrintProfilingStats
from build import build
from producer_pool import ProducerPool
import batch_provider_tree

def set_verbose(verbose=True):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
