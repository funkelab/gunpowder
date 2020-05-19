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

        if (
                spec is not None and
                spec.roi is not None and
                spec.voxel_size is not None):

            for d in range(len(spec.voxel_size)):
                assert spec.voxel_size[d]*data.shape[-spec.roi.dims()+d] == spec.roi.get_shape()[d], \
                        "ROI %s does not align with voxel size %s * data shape %s"%(spec.roi, spec.voxel_size, data.shape)
                if spec.roi.get_offset()[d] is not None:
                    assert spec.roi.get_offset()[d] % spec.voxel_size[d] == 0,\
                            "ROI offset %s must be a multiple of voxel size %s"\
                            % (spec.roi.get_offset(), spec.voxel_size)

        if spec.dtype is not None:
            assert data.dtype == spec.dtype, \
                "data dtype %s does not match spec dtype %s" % (data.dtype, spec.dtype)

        self.freeze()

    def crop(self, roi, copy=True):
        '''Create a cropped copy of this Array.

        Args:

            roi (:class:`Roi`):

                ROI in world units to crop to.

            copy (``bool``):

                Make a copy of the data.
        '''

        assert self.spec.roi.contains(roi), (
            "Requested crop ROI (%s) doesn't fit in array (%s)" %
            (roi, self.spec.roi))

        if self.spec.roi == roi and not copy:
            return self

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

    def merge(self, array, copy_from_self=False, copy=False):
        '''Merge this array with another one. The resulting array will have the
        size of the larger one, with values replaced from ``array``.

        This only works if one of the two arrays is contained in the other. In
        this case, ``array`` will overwrite values in ``self`` (unless
        ``copy_from_self`` is set to ``True``).

        A copy will only be made if necessary or ``copy`` is set to ``True``.
        '''
        # It is unclear how to merge arrays in all cases. Consider a 10x10 array,
        # you crop out a 5x5 area, do a shift augment, and attempt to merge.
        # What does that mean? specs have changed. It should be a new key.
        raise NotImplementedError("Merge function should not be used!")

        self_roi = self.spec.roi
        array_roi = array.spec.roi

        assert self_roi.contains(array_roi) or array_roi.contains(self_roi), \
            "Can not merge arrays that are not contained in each other."

        assert self.spec.voxel_size == array.spec.voxel_size, \
            "Can not merge arrays with different voxel sizes."

        # make sure self contains array
        if not self_roi.contains(array_roi):
            return array.merge(self, not copy_from_self, copy)

        # -> here we know that self contains array

        # simple case, self overwrites all of array
        if copy_from_self:
            return self if not copy else deepcopy(self)

        # -> here we know that copy_from_self == False

        # simple case, ROIs are the same
        if self_roi == array_roi:
            return array if not copy else deepcopy(array)

        # part of self have to be replaced, a copy is needed
        merged = deepcopy(self)

        voxel_size = self.spec.voxel_size
        data_roi = (array_roi - self_roi.get_offset())/voxel_size
        slices = data_roi.get_bounding_box()

        while len(slices) < len(self.data.shape):
            slices = (slice(None),) + slices

        merged.data[slices] = array.data

        return merged

    def __repr__(self):
        return str(self.spec)

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
