import logging

import batch_provider_tree
from add_gt_affinities import AddGtAffinities
from batch import Batch
from batch_filter import BatchFilter
from batch_spec import BatchSpec
from build import build
from chunk import Chunk
from coordinate import Coordinate
from defect_augment import DefectAugment
from elastic_augmentation import ElasticAugmentation
from exclude_labels import ExcludeLabels
from grow_boundary import GrowBoundary
from hdf5_source import Hdf5Source
from intensity_augment import IntensityAugment
from intensity_scale_shift import IntensityScaleShift
from normalize import Normalize
from padding import Padding
from precache import PreCache
from print_profiling_stats import PrintProfilingStats
from producer_pool import ProducerPool
from random_location import RandomLocation
from random_provider import RandomProvider
from reject import Reject
from roi import Roi
from simple_augment import SimpleAugment
from snapshot import Snapshot
from split_and_renumber_segmentation_labels import SplitAndRenumberSegmentationLabels
from zero_out_const_sections import ZeroOutConstSections

logging.basicConfig(level=logging.INFO)


def set_verbose(verbose=True):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
