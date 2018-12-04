from __future__ import absolute_import

from .add_affinities import TestAddAffinities
from .add_boundary_distance_gradients import TestAddBoundaryDistanceGradients
from .add_vector_map import TestAddVectorMap
from .balance_labels import TestBalanceLabels
from .crop import TestCrop
from .downsample import TestDownSample
from .dvid_source import TestDvidSource
from .elastic_augment_points import TestElasticAugment
from .hdf5_source import TestHdf5Source, TestN5Source, TestZarrSource
from .hdf5_write import TestHdf5Write
from .shift_augment import TestShiftAugment2D
from .merge_provider import TestMergeProvider
from .n5_write import TestN5Write
from .normalize import TestNormalize
from .pad import TestPad
from .points_keys import TestPointsKeys
from .precache import TestPreCache
from .prepare_malis import TestPrepareMalis
from .profiling import TestProfiling
from .provider_test import ProviderTest
from .random_location import TestRandomLocation
from .random_location_points import TestRandomLocationPoints
from .rasterize_points import TestRasterizePoints
from .scan import TestScan
from .tensorflow_train import TestTensorflowTrain
from .zarr_write import TestZarrWrite
