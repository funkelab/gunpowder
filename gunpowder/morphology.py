import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

def enlarge_binary_map(
    binary_map,
    ball_radius_voxel=1,
    voxel_size=None,
    ball_radius_physical=None,
    sphere_inner_radius=None):
    '''Enlarge existing regions in a binary map.

    Args:

        binary_map (numpy array):

            A matrix with zeros, in which regions to be enlarged are indicated
            with a 1 (regions can already represent larger areas).

        ball_radius_voxel (int):

            Enlarged region have a marker_size (measured in voxels) margin
            added to the already existing region (taking into account the
            provided voxel_size). For instance a ball_radius_voxel of 1 and a
            voxel_size of [2, 1, 1] (z, y, x) would add a voxel margin of 1 in
            x,y-direction and no margin in z-direction.

        voxel_size (tuple, list or numpy array):

            Indicates the physical voxel size of the binary_map.

        ball_radius_physical (int):

            If set, overwrites the ball_radius_voxel parameter. Provides the
            margin size in physical units. For instance, a voxel_size of [20,
            10, 10] and ball_radius_physical of 10 would add a voxel margin of
            1 in x,y-direction and no margin in z-direction.

    Returns:

        A matrix with 0s and 1s of same dimension as input binary_map with
        enlarged regions (indicated with 1)
    '''
    if len(np.unique(binary_map)) == 1:
        # Check whether there are regions at all. If  there is no region (or everything is full), return the same map.
        return binary_map
    if voxel_size is None:
        voxel_size = (1,)*binary_map.shape[0]
    voxel_size = np.asarray(voxel_size)
    if ball_radius_physical is None:
        voxel_size /= np.min(voxel_size)
        marker_size = ball_radius_voxel
    else:
        marker_size = ball_radius_physical
    binary_map = np.logical_not(binary_map)
    edtmap = distance_transform_edt(binary_map, sampling=voxel_size)
    binary_map = edtmap <= marker_size
    if sphere_inner_radius is not None:
        binary_map[edtmap <= sphere_inner_radius] = False
    binary_map = binary_map.astype(np.uint8)
    return binary_map
