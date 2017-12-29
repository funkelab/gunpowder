from .freezable import Freezable
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Points(Freezable):
    '''A list of :class:``Point``s with a specification describing the data.

    Args:

        data (dict, int->Point): A dictionary of IDs mapping to
            :class:``Point``s.

        spec (:class:`PointsSpec`): A spec describing the data.
    '''

    def __init__(self, data, spec):
        self.data = data
        self.spec = spec
        self.freeze()

class Point(Freezable):
    def __init__(self, location):
        self.location = location
        self.freeze()

class PointsKey(Freezable):
    '''A key to identify lists of points in requests, batches, and across
    nodes.

    Used as key in :class:``BatchRequest`` and :class:``Batch`` to retrieve
    specs or lists of points.

    Args:

        identifier (string):
            A unique, human readable identifier for this points key. Will be
            used in log messages and to look up points in requests and batches.
            Should be upper case (like ``CENTER_POINTS``). The identifier is
            unique: Two points keys with the same identifier will refer to the
            same points.
    '''

    def __init__(self, identifier):
        self.identifier = identifier
        self.hash = hash(identifier)
        self.freeze()
        logger.debug("Registering points type %s", self)
        setattr(PointsKeys, self.identifier, self)

    def __eq__(self, other):
        return hasattr(other, 'identifier') and self.identifier == other.identifier

    def __hash__(self):
        return self.hash

    def __repr__(self):
        return self.identifier

class PointsKeys:
    '''Convenience access to all created :class:``PointsKey``s. A key generated
    with::

        centers = PointsKey('CENTER_POINTS')

    can be retrieved as::

        PointsKeys.CENTER_POINTS
    '''
    pass

class RasterizationSetting(Freezable):
    '''Data structure to store parameters for rasterization of points.

    Args:

        marker_size_voxel (int):

            Parameter only used, when ``marker_size_physical`` is not set/set
            to None. Specifies the blob radius in voxel units.

        marker_size_physical (int):

            If set, overwrites the marker_size_voxel parameter. Provides the
            radius size in physical units. For instance, a points resolution of
            [20, 10, 10] and marker_size_physical of 10 would create a blob
            with a radius of 1 in x,y-direction and no radius in z-direction.

        stay_inside_arraytype (:class:``ArrayKey``):

            Used to mask out created blobs. The array is assumed to contain
            discrete objects. The object id at the specific point being
            rasterized is used to crop the blob. Blob regions that are located
            outside of the object are masked out, such that the blob is only
            inside the specific object.

        donut_inner_radius (int):

            If set, instead of a blob, a donut is created. The size of the
            whole donut corresponds to the size specified with
            marker_size_physical or marker_size_voxel. The size of the inner
            radius (the region being cropped out) corresponds to
            donut_inner_radius. This parameter has to be provided in the same
            unit as the specified marker_size.

        voxel_size (:class:``Coordinate``, optional):

            The voxel size of the array to create in world units.
    '''
    def __init__(
            self,
            marker_size_voxel=1,
            marker_size_physical=None,
            stay_inside_arraytype=None,
            donut_inner_radius=None,
            voxel_size=None,
            invert_map=False):

        if donut_inner_radius is not None:
            if marker_size_physical is not None:
                marker_size_check = marker_size_physical
            else:
                marker_size_check = marker_size_voxel
            assert donut_inner_radius < marker_size_check, (
                "trying to create a donut in which the inner radius is larger "
                "than the donut size")
        self.marker_size_voxel = marker_size_voxel
        self.marker_size_physical = marker_size_physical
        self.stay_inside_arraytype = stay_inside_arraytype
        self.donut_inner_radius = donut_inner_radius
        self.voxel_size = voxel_size
        self.invert_map = invert_map
        self.freeze()
