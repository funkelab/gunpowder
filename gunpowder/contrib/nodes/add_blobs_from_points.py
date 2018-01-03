import logging
import itertools
import numpy as np

from gunpowder.array import Array
from gunpowder.nodes.batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class AddBlobsFromPoints(BatchFilter):
    '''Add an array with blobs at locations given by a specified points
    collection. The blobs are also restricted to stay within the same class in
    the restrictive_mask array that corresponds to the center voxel of the
    blob.

    Args:

        blob_settings(dict):

            Where each desired output blob map should have it's own entry
            consisting of the following format:

            `blob_name` : dict (

                'points_key' : Desired point type to use for blob locations

                'output_array_key': Desired array type name for output map

                'output_array_dtype': Desired array dtype name for output map

                'radius': Desired radius of blobs, since blobs are restricted
                by the restricting mask, this radius should be thought of as
                the maximum radius of the blobs.

                'output_voxel_size': voxel_size of output array. Voxel size of
                restrictive mask

                'restrictive_mask_key': Array type of restrictive mask

                'id_mapper': Functor (class with a __call__ function) that can
                take an ID and map it to some other value. This class should
                also have a 'make_map' method that will be called at the
                beggining of each process step and given all ids in all arrays
                to be processed for that batch.
            )

            This is an example blob_setting for presynaptic blobs in the cremi
            dataset::

              add_blob_data = {
                'PRESYN': {
                  'points_key': PointsTypes.PRESYN,
                  'output_array_key': ArrayTypes.PRESYN_BLOB,
                  'output_array_dtype': 'int64',
                  'radius': 60,
                  'output_voxel_size': voxel_size,
                  'restrictive_mask_key': ArrayTypes.GT_LABELS,
                  'max_desired_overlap': 0.05
                }
              }
    '''

    def __init__(self, blob_settings):

        self.blob_settings = blob_settings

        for points_key, settings in self.blob_settings.items():
            blob_settings[points_key]['blob_placer'] = BlobPlacer(
                radius=settings['radius'],
                voxel_size=settings['output_voxel_size'],
                dtype=settings['output_array_dtype']
                )

    def setup(self):
        for blob_name, settings in self.blob_settings.items():
            self.provides(
                settings['output_array_key'],
                self.spec[settings['restrictive_mask_key']]
                )

    def prepare(self, request):

        for blob_name, settings in self.blob_settings.items():
            array_key = settings['output_array_key']
            if array_key in request:

                points_key = settings['points_key']
                request_roi = request[array_key].roi


                # If point is not already requested, add to request
                if points_key not in request.points_specs:
                    request.add(points_key, request_roi.get_shape())
                else:
                    request[points_key].roi =\
                     request[points_key].roi.union(request_roi)

                # Get correct size for restrictive_mask_key
                restrictive_mask_key = settings['restrictive_mask_key']
                if restrictive_mask_key not in request.array_specs:
                    request.add(restrictive_mask_key, request_roi.get_shape())
                else:
                    request[restrictive_mask_key].roi =\
                     request[restrictive_mask_key].roi.union(request_roi)
            else:
                # do nothing if no blobs of this type were requested
                logger.warning('%s output array type for %s never requested. \
                    Deleting entry...'%(settings['output_array_key'], blob_name))
                del self.blob_settings[blob_name]


    def process(self, batch, request):

        # check arrays and gather all IDs and synapse IDs
        all_points = {}
        all_synapse_ids = {}

        for blob_name, settings in self.blob_settings.items():
            # Unpack settings
            points_key = settings['points_key']
            restrictive_mask_key = settings['restrictive_mask_key']


            # Make sure both the necesary point types and arrays are present
            assert points_key in batch.points, "Upstream does not provide required point type\
            : %s"%points_key

            assert restrictive_mask_key in batch.arrays, "Upstream does not provide required \
            array type: %s"%restrictive_mask_key

            # Get point data
            points = batch.points[points_key]

            # If point doesn't have it's corresponding partner, delete it
            if 'partner_points' in settings.keys() and settings['partner_points'] is not None:
                partner_points = batch.points[settings['partner_points']]
                synapse_ids = []
                for point_id, point in points.data.items():
                    # pdb.set_trace()
                    if not point.partner_ids[0] in partner_points.data.keys():
                        logger.warning('Point %s has no partner. Deleting...'%point_id)
                        del points.data[point_id]
                    else:
                        synapse_ids.append(point.synapse_id)

            all_synapse_ids[points_key] = synapse_ids
            all_points[points_key] = points

        for blob_name, settings in self.blob_settings.items():

            # Unpack settings
            points_key = settings['points_key']
            array_key = settings['output_array_key']
            voxel_size = settings['output_voxel_size']
            restrictive_mask_key = settings['restrictive_mask_key']
            restrictive_mask = batch.arrays[restrictive_mask_key].crop(request[array_key].roi)

            id_mapper = settings['id_mapper']
            dtype = settings['output_array_dtype']

            if id_mapper is not None:
                id_mapper.make_map(all_points)

            # Initialize output array
            shape_array = np.asarray(request[array_key].roi.get_shape())/voxel_size
            blob_map = np.zeros(shape_array, dtype=dtype)

            # Get point data
            points = batch.points[points_key]


            offset = np.asarray(points.spec.roi.get_offset())
            for point_id, point_data in points.data.items():
                voxel_location = np.round(((point_data.location - offset)/(voxel_size))).astype('int32')

                synapse_id = point_data.synapse_id
                # if mapping exists, do it
                if id_mapper is not None:
                    synapse_id = id_mapper(synapse_id)

                settings['blob_placer'].place(blob_map, voxel_location,
                    synapse_id, restrictive_mask.data)


            # Provide array
            batch.arrays[array_key] = Array(blob_map, spec=request[array_key].copy())
            batch.arrays[array_key].spec.dtype = dtype

            # add id_mapping to attributes
            if id_mapper is not None:
                id_map_list = np.array(list(id_mapper.get_map().items()))
                batch.arrays[array_key].attrs['id_mapping'] = id_map_list

            batch.arrays[array_key].attrs['point_ids'] = points.data.keys()
            batch.arrays[array_key].attrs['synapse_ids'] = all_synapse_ids[points_key]

            # Crop all other requests
        for array_key, array in request.array_specs.items():
            batch.arrays[array_key] = batch.arrays[array_key].crop(array.roi)

        for points_key, points in request.points_specs.items():
            batch.points[points_key] = batch.points[points_key].spec.roi = points.roi

class BlobPlacer:
    ''' Places synapse array blobs from location data.
        Args:
            radius: int - that desired radius of synaptic blobs
            voxel_size: array, list, tuple - voxel size in physical
        '''

    def __init__(self, radius, voxel_size, dtype='uint64'):

        self.voxel_size = voxel_size
        if isinstance(self.voxel_size, (list, tuple)):
            self.voxel_size = np.asarray(self.voxel_size)

        self.radius = (radius/self.voxel_size)
        self.sphere_map = np.zeros(self.radius*2, dtype=dtype)
        self.center = (np.asarray(self.sphere_map.shape))/2

        ranges = [range(0, self.radius[0]*2),
                  range(0, self.radius[1]*2),
                  range(0, self.radius[2]*2)]

        for index in np.asarray(list(itertools.product(*ranges))):
            # if distance less than r, place a 1
            if np.linalg.norm((self.center - index)*self.voxel_size) <= radius:
                self.sphere_map[tuple(index)] = 1

        self.sphere_voxel_array = np.sum(self.sphere_map, axis=(0, 1, 2))

    def place(self, matrix, location, marker, mask):
        ''' Places synapse
        Args:
            matrix: 4D np array - 1st dim are for layers to avoid overlap
            (3 should be more than enough)
            location: np array - location where to place synaptic blob within given matrix
            marker: int - the ID used to mark this paricular synapse in the matrix
            mask:   3D np array - when placing a blob, will sample mask at
        center location and only place blob in interection where mask has
        the same ID. Usually used to restrict synaptic blobs inside their
        respective cells (using segmentation)
        '''
        # Calculate cube circumscribing the sphere to place
        start = location - self.radius
        end = location + self.radius

        # check if sphere fits in matrix
        if np.all(start >= 0) and np.all(np.asarray(matrix.shape) - end >= 0):

            # calculate actual synapse shape from intersection between sphere and restrictive mask
            restricting_label = mask[location[0], location[1], location[2]]

            restricting_mask = \
            mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]] == restricting_label

            shape = (self.sphere_map*restricting_mask)

            # place shape in chosen layer
            matrix[start[0]:end[0], start[1]:end[1], start[2]:end[2]] += shape*marker
            return matrix, True

        logger.warning('Location %s out of bounds'%(location))
        return matrix, False
