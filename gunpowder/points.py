from .freezable import Freezable
import logging
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

logger = logging.getLogger(__name__)

class PointsType:
    '''Describes general properties of a points type.

    Args:

        identifier (string):
            A human readable identifier for this points type. Will be used as a 
            static attribute in :class:`PointsTypes`. Should be upper case (like 
            ``PRESYN``, ``POSTSYN``).

    '''

    def __init__(self, identifier):
        self.identifier = identifier
        self.hash = hash(identifier)

    def __eq__(self, other):
        return hasattr(other, 'identifier') and self.identifier == other.identifier

    def __hash__(self):
        return self.hash

    def __repr__(self):
        return self.identifier

class PointsTypes:
    '''An expandable collection of points types, which initially contains:

        ===================  ====================================================
        identifier           purpose
        ===================  ====================================================
        ``PRESYN``           Presynaptic locations
        ``POSTSYN``          Postsynaptic locations
        ===================  ====================================================

    New points types can be added with :func:`register_points_type`.
    '''
    pass

def register_points_type(identifier):
    '''Register a new points type.

    For example, the following call::

            register_points_type('IDENTIFIER')

    will create a new points type available as ``PointsTypes.IDENTIFIER``.
    ``PointsTypes.IDENTIFIER`` can then be used in dictionaries, as it is done
    in :class:`BatchRequest` and :class:`ProviderSpec`, for example.
    '''
    points_type = PointsType(identifier)
    logger.debug("Registering points type " + str(points_type))
    setattr(PointsTypes, points_type.identifier, points_type)

register_points_type('PRESYN')
register_points_type('POSTSYN')


class Points(Freezable):
    def __init__(self, data, spec):
        """ Data structure to keep information about points locations within a ROI
        :param data:        a dictionary with node_ids as keys and Point instances as values
        :param spec:        A :class:`PointsSpec` describing the metadata of the points
        """
        self.data = data
        self.spec = spec

        self.freeze()


class Point(Freezable):
    def __init__(self, location):
        self.location = location

        self.freeze()

class PreSynPoint(Point):
    def __init__(self, location, location_id, synapse_id, partner_ids, props=None):
        """ Presynaptic locations
        :param location:     ndarray, [zyx]
        :param location_id:  int, unique for every synapse location across pre and postsynaptic locations 
        :param synapse_id:   int, unique for every synapse(synaptic partners have the same synapse_id, but different location_ids)
        :param partner_ids:  list of ints, location ids of postsynaptic partners
        :param props:        dict, properties
        """
        Point.__init__(self, location=location)
        self.thaw()

        self.location_id  = location_id
        self.synapse_id   = synapse_id
        self.partner_ids  = partner_ids
        if props is None:
            self.props = {}
        else:
            self.props = props

        self.freeze()

class PostSynPoint(Point):
    def __init__(self, location, location_id, synapse_id, partner_ids, props=None):
        """
        :param location:     ndarray, [zyx]
        :param location_id:  int, unique for every synapse location across pre and postsynaptic locations 
        :param synapse_id:   int, unique for every synapse(synaptic partners have the same synapse_id, but different location_ids)
        :param partner_ids:  list of int, location id of presynaptic partner
        :param props:        dict, properties
        """
        Point.__init__(self, location=location)
        self.thaw()

        self.location_id  = location_id
        self.synapse_id   = synapse_id
        self.partner_ids  = partner_ids
        if props is None:
            self.props = {}
        else:
            self.props = props

        self.freeze()


def enlarge_binary_map(binary_map, marker_size_voxel=1, voxel_size=None, marker_size_physical=None,
                       donut_inner_radius=None):
    """
    Enlarge existing regions in a binary map.
    Parameters
    ----------
        binary_map: numpy array
            matrix with zeros, in which regions to be enlarged are indicated with a 1 (regions can already
            represent larger areas)
        marker_size_voxel: int
            enlarged region have a marker_size (measured in voxels) margin added to
            the already existing region (taking into account the provided voxel_size). For instance a marker_size_voxel
            of 1 and a voxel_size of [2, 1, 1] (z, y, x) would add a voxel margin of 1 in x,y-direction and no margin
            in z-direction.
        voxel_size:     tuple, list or numpy array
            indicates the physical voxel size of the binary_map.
        marker_size_physical: int
            if set, overwrites the marker_size_voxel parameter. Provides the margin size in physical units. For
            instance, a voxel_size of [20, 10, 10] and marker_size_physical of 10 would add a voxel margin of 1 in
            x,y-direction and no margin in z-direction.
    Returns
    ---------
        binary_map: matrix with 0s and 1s of same dimension as input binary_map with enlarged regions (indicated with 1)
    """
    if len(np.unique(binary_map)) == 1:
        # Check whether there are regions at all. If  there is no region (or everything is full), return the same map.
        return binary_map
    if voxel_size is None:
        voxel_size = (1,)*binary_map.shape[0]
    voxel_size = np.asarray(voxel_size)
    if marker_size_physical is None:
        voxel_size /= np.min(voxel_size)
        marker_size = marker_size_voxel
    else:
        marker_size = marker_size_physical
    binary_map = np.logical_not(binary_map)
    edtmap = distance_transform_edt(binary_map, sampling=voxel_size)
    binary_map = edtmap <= marker_size
    if donut_inner_radius is not None:
        binary_map[edtmap <= donut_inner_radius] = False
    binary_map = binary_map.astype(np.uint8)
    return binary_map


class RasterizationSetting(Freezable):
    '''Data structure to store parameters for rasterization of points.
    Args:
        marker_size_voxel (int): parameter only used, when marker_size_physical is not set/set to None. Specifies the
        blob radius in voxel units.

        marker_size_physical (int): if set, overwrites the marker_size_voxel parameter. Provides the radius size in
        physical units. For instance, a points resolution of [20, 10, 10] and marker_size_physical of 10 would create a
        blob with a radius of 1 in x,y-direction and no radius in z-direction.

        stay_inside_volumetype (Volume.VolumeType): specified volume is used to mask out created blobs. The volume is
        assumed to contain discrete objects. The object id at the specific point being rasterized is used to crop the
        blob. Blob regions that are located outside of the object are masked out, such that the blob is only inside the
        specific object.

        donut_inner_radius (int) : if set, instead of a blob, a donut is created. The size of the whole donut
        corresponds to the size specified with marker_size_physical or marker_size_voxel. The size of the inner radius
        (the region being cropped out) corresponds to donut_inner_radius. This parameter has to be provided in the same
        unit as the specified marker_size.

    Notes:
        Takes the resolution provided for the respective points into account. Eg. anistropic resolutions result in
        anistropict blob creations, as expected.
    '''
    def __init__(self, marker_size_voxel=1, marker_size_physical=None,
                 stay_inside_volumetype=None, donut_inner_radius=None, invert_map=False):
        self.thaw()
        if donut_inner_radius is not None:
            if marker_size_physical is not None:
                marker_size_check = marker_size_physical
            else:
                marker_size_check = marker_size_voxel
            assert donut_inner_radius < marker_size_check, 'trying to create a donut in which the inner ' \
                                                              'radius is larger than the donut size'
        self.marker_size_voxel = marker_size_voxel
        self.marker_size_physical = marker_size_physical
        self.stay_inside_volumetype = stay_inside_volumetype
        self.donut_inner_radius = donut_inner_radius
        self.invert_map = invert_map
        self.freeze()








