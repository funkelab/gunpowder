import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

def enlarge_binary_map(
    binary_map,
    radius,
    voxel_size,
    ring_inner=None):
    '''Enlarge existing regions in a binary map.

    Args:

        binary_map (numpy array):

            A matrix with zeros, in which regions to be enlarged are indicated
            with a 1 (regions can already represent larger areas).

        radius (int):

            Enlarged region have a marker_size (measured in voxels) margin
            added to the already existing region (taking into account the
            provided voxel_size). For instance a radius of 1 and a voxel_size
            of [2, 1, 1] (z, y, x) would add a voxel margin of 1 in
            x,y-direction and no margin in z-direction.

        voxel_size (tuple, list or numpy array):

            Indicates the physical voxel size of the binary_map.

        ring_inner (int, optional):

            If set, instead of just enlargin objects, a ring is grown around
            them (and the objects removed). The ring starts at ``ring_inner``
            and goes until ``radius``.

    Returns:

        A matrix with 0s and 1s of same dimension as input binary_map with
        enlarged regions (indicated with 1)
    '''

    if len(np.unique(binary_map)) == 1:
        # Check whether there are regions at all. If there is no region (or
        # everything is full), return the same map.
        return binary_map

    if voxel_size is None:
        voxel_size = (1,)*binary_map.shape[0]

    voxel_size = np.asarray(voxel_size)

    binary_map = np.logical_not(binary_map)
    edtmap = distance_transform_edt(binary_map, sampling=voxel_size)

    # grow objects
    binary_map = edtmap <= radius

    # unmask inner part, if requested
    if ring_inner is not None:
        binary_map[edtmap <= ring_inner] = False

    return binary_map
