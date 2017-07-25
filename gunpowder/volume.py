from .freezable import Freezable
from gunpowder.coordinate import Coordinate
import copy
import logging

logger = logging.getLogger(__name__)

class VolumeType(Freezable):
    '''Describes general properties of a volume type.

    Args:

        identifier (string):
            A human readable identifier for this volume type. Will be used as a 
            static attribute in :class:`VolumeTypes`. Should be upper case (like 
            ``RAW``, ``GT_LABELS``).

        interpolate (bool):
            Indicates whether voxels can be interpolated (as for intensities) or 
            not (as for labels). This will be used by nodes that perform data 
            augmentations.

        voxel_size (tuple of int):
            The size of a voxel in world units.
    '''

    def __init__(self, identifier, interpolate, voxel_size=(1,1,1)):
        self.identifier = identifier
        self.interpolate = interpolate
        self.voxel_size = Coordinate(voxel_size)
        self.hash = hash(identifier)
        self.freeze()

    def __eq__(self, other):
        return hasattr(other, 'identifier') and self.identifier == other.identifier

    def __hash__(self):
        return self.hash

    def __repr__(self):
        return self.identifier

class VolumeTypes:
    '''An expandable collection of volume types, which initially contains:

        =================================  ====================================================
        identifier                         purpose
        =================================  ====================================================
        ``RAW``                            Raw intensity volumes.
        ``ALPHA_MASK``                     Alpha mask for blending
                                           raw volumes
                                           (used in :class:`DefectAugment`).
        ``GT_LABELS``                      Ground-truth object IDs.
        ``GT_AFFINITIES``                  Ground-truth affinities.
        ``GT_MASK``                        Binary mask (1-use, 0-don't use) on ground-truth. No 
                                           assumptions about masked out area (i.e., end of 
                                           ground-truth).
        ``GT_IGNORE``                      Binary mask (1-use, 0-don't use) on ground-truth. 
                                           Assumes that transition between 0 and 1 lies on an 
                                           object boundary.
        ``PRED_AFFINITIES``                Predicted affinities.
        ``LOSS_SCALE``                     Used for element-wise multiplication with loss for
                                           training.
        ``LOSS_GRADIENT``                  Gradient of the training loss.
        ``GT_BM_PRESYN``                   Ground truth of binary map for presynaptic locations
        ``GT_BM_PRESYN``                   Ground truth of binary map for postsynaptic locations
        ``GT_MASK_EXCLUSIVEZONE_PRESYN``   ExculsiveZone binary mask (1-use, 
                                           0-don't use) around presyn locations
        ``GT_MASK_EXCLUSIVEZONE_POSTSYN``  ExculsiveZone binary mask (1-use, 
                                           0-don't use) around postsyn locations
        ``PRED_BM_PRESYN``                 Predicted presynaptic locations
        ``PRED_BM_POSTSYN``                Predicted postsynaptic locations
        =================================  ====================================================

    New volume types can be added with :func:`register_volume_type`.
    '''
    pass

def register_volume_type(volume_type):
    '''Register a new volume type.

    For example, the following call::

            register_volume_type(VolumeType('IDENTIFIER', interpolate=True))

    will create a new volume type available as ``VolumeTypes.IDENTIFIER``. 
    ``VolumeTypes.IDENTIFIER`` can then be used in dictionaries, as well as 
    being queried for further specs like ``VolumeType.interpolate``.
    '''
    logger.debug("Registering volume type " + str(volume_type))
    setattr(VolumeTypes, volume_type.identifier, volume_type)

register_volume_type(VolumeType('RAW', interpolate=True))
register_volume_type(VolumeType('ALPHA_MASK', interpolate=True))
register_volume_type(VolumeType('GT_LABELS', interpolate=False))
register_volume_type(VolumeType('GT_AFFINITIES', interpolate=False))
register_volume_type(VolumeType('GT_MASK', interpolate=False))
register_volume_type(VolumeType('GT_IGNORE', interpolate=False))
register_volume_type(VolumeType('PRED_AFFINITIES', interpolate=False))
register_volume_type(VolumeType('LOSS_SCALE', interpolate=False))
register_volume_type(VolumeType('LOSS_GRADIENT', interpolate=False))

register_volume_type(VolumeType('GT_BM_PRESYN', interpolate=False))
register_volume_type(VolumeType('GT_BM_POSTSYN', interpolate=False))
register_volume_type(VolumeType('GT_MASK_EXCLUSIVEZONE_PRESYN', interpolate=False))
register_volume_type(VolumeType('GT_MASK_EXCLUSIVEZONE_POSTSYN', interpolate=False))
register_volume_type(VolumeType('PRED_BM_PRESYN', interpolate=False))
register_volume_type(VolumeType('PRED_BM_POSTSYN', interpolate=False))
register_volume_type(VolumeType('LOSS_GRADIENT_PRESYN', interpolate=False))
register_volume_type(VolumeType('LOSS_GRADIENT_POSTSYN', interpolate=False))

class Volume(Freezable):

    def __init__(self, data, roi):

        self.roi = roi
        self.data = data

        voxel_size = self.get_voxel_size()
        for d in range(len(voxel_size)):
            assert voxel_size[d]*data.shape[-self.roi.dims()+d] == roi.get_shape()[d], \
                    "ROI %s does not align with voxel size %s * data shape %s"%(roi, voxel_size, data.shape)

        self.freeze()

    def get_voxel_size(self):

        return self.roi.get_shape()/self.data.shape[-self.roi.dims():]

    def crop(self, roi, copy=True):
        '''Create a cropped copy of this Volume.

        Args:

            roi(:class:``Roi``): ROI in world units to crop to.

            copy(bool): Make a copy of the data (default).
        '''

        assert self.roi.contains(roi)

        voxel_size = self.get_voxel_size()
        data_roi = (roi - self.roi.get_offset())/voxel_size
        slices = data_roi.get_bounding_box()

        while len(slices) < len(self.data.shape):
            slices = (slice(None),) + slices

        data = self.data[slices]
        if copy:
            data = np.array(data)
        return Volume(data, copy.deepcopy(roi))
