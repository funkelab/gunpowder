from .freezable import Freezable
from copy import deepcopy
from gunpowder.coordinate import Coordinate
from gunpowder.roi import Roi
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Array(Freezable):
    '''A numpy array with a specification describing the data.

    Args:

        data (array-like):

            The data to be stored in the array. Will be converted to a numpy
            array, if necessary.

        spec (:class:`ArraySpec`, optional):

            A spec describing the data.

        attrs (``dict``, optional):

            Optional attributes to describe this array.
    '''

    def __init__(self, data, spec=None, attrs=None):

        self.spec = deepcopy(spec)
        self.data = np.asarray(data)
        self.attrs = attrs

        if attrs is None:
            self.attrs = {}

        if spec is not None and spec.roi is not None:
            for d in range(len(spec.voxel_size)):
                assert spec.voxel_size[d]*data.shape[-spec.roi.dims()+d] == spec.roi.get_shape()[d], \
                        "ROI %s does not align with voxel size %s * data shape %s"%(spec.roi, spec.voxel_size, data.shape)

        self.freeze()

    def crop(self, roi, copy=True):
        '''Create a cropped copy of this Array.

        Args:

            roi(:class:`Roi`):

                ROI in world units to crop to.

            copy(``bool``):

                Make a copy of the data (default).
        '''

        assert self.spec.roi.contains(roi), "Requested crop ROI (%s) doesn't fit in array (%s)"\
        %(roi, self.spec.roi)

        voxel_size = self.spec.voxel_size
        data_roi = (roi - self.spec.roi.get_offset())/voxel_size
        slices = data_roi.get_bounding_box()

        while len(slices) < len(self.data.shape):
            slices = (slice(None),) + slices

        data = self.data[slices]
        if copy:
            data = np.array(data)

        spec = deepcopy(self.spec)
        attrs = deepcopy(self.attrs)
        spec.roi = deepcopy(roi)
        return Array(data, spec, attrs)

class ArrayKey(Freezable):
    '''A key to identify arrays in requests, batches, and across nodes.

    Used as key in :class:`BatchRequest` and :class:`Batch` to retrieve array
    specs or arrays.

    Args:

        identifier (``string``):

            A unique, human readable identifier for this array key. Will be
            used in log messages and to look up arrays in requests and batches.
            Should be upper case (like ``RAW``, ``GT_LABELS``). The identifier
            is unique: Two array keys with the same identifier will refer to
            the same array.
    '''

    def __init__(self, identifier):
        self.identifier = identifier
        self.hash = hash(identifier)
        self.freeze()
        logger.debug("Registering array key %s", self)
        setattr(ArrayKeys, self.identifier, self)

    def __eq__(self, other):
        return hasattr(other, 'identifier') and self.identifier == other.identifier

    def __hash__(self):
        return self.hash

    def __repr__(self):
        return self.identifier

class ArrayKeys:
    '''Convenience access to all created :class:``ArrayKey``s. A key generated
    with::

        raw = ArrayKey('RAW')

    can be retrieved as::

        ArrayKeys.RAW
    '''
    pass
