import logging
import itertools
import numpy as np

from gunpowder.volume import Volume
from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class AddBlobsFromPoints(BatchFilter):
    '''Add a volume with blobs at locations given by a specified point type.
    The blobs are also restricted to stay within the same class in the restrictive_mask volume
    that corresponds to the center voxel of the blob.

    Args:

        blob_settings(dict) :   Where each desired output blob map should have it's own entry
        consisting of the following format:

        `blob_name` : dict (
            'point_type' : Desired point type to use for blob locations
            'output_volume_type': Desired volume type name for output map
            'output_volume_dtype': Desired volume dtype name for output map
            'radius': Desired radius of blobs, since blobs are restricted by the restricting mask,
                      this radius should be thought of as the maximum radius of the blobs.
            'output_voxel_size': voxel_size of output volume. Voxel size of restrictive mask
            'restrictive_mask_type': Volume type of restrictive mask
            'id_mapper': Functor (class with a __call__ function) that can take an ID and map it to
            some other value. This class should also have a 'make_map' method that will be
            called at the beggining of each process step and given all ids in all volumes to be
            processed for that batch.
        )

        This is an example blob_setting for presynaptic blobs in the cremi dataset:

        add_blob_data = {
                        'PRESYN':
                            {
                                'point_type': PointsTypes.PRESYN,
                                'output_volume_type': VolumeTypes.PRESYN_BLOB,
                                'output_volume_dtype': 'int64',
                                'radius': 60,
                                'output_voxel_size': voxel_size,
                                'restrictive_mask_type': VolumeTypes.GT_LABELS,
                                'max_desired_overlap': 0.05
                            }
                        }
    '''

    def __init__(self, blob_settings):

        self.blob_settings = blob_settings

        for point_type, settings in self.blob_settings.items():
            blob_settings[point_type]['blob_placer'] = BlobPlacer(
                radius=settings['radius'],
                voxel_size=settings['output_voxel_size'],
                dtype=settings['output_volume_dtype']
                )

    def setup(self):
        for blob_name, settings in self.blob_settings.items():
            self.provides(
                settings['output_volume_type'],
                self.spec[settings['restrictive_mask_type']]
                )

    def prepare(self, request):

        for blob_name, settings in self.blob_settings.items():
            volume_type = settings['output_volume_type']
            if volume_type in request:

                point_type = settings['point_type']
                request_roi = request[volume_type].roi


                # If point is not already requested, add to request
                if point_type not in request.points_specs:
                    request.add(point_type, request_roi.get_shape())
                else:
                    request[point_type].roi =\
                     request[point_type].roi.union(request_roi)

                # Get correct size for restrictive_mask_type
                restrictive_mask_type = settings['restrictive_mask_type']
                if restrictive_mask_type not in request.volume_specs:
                    request.add(restrictive_mask_type, request_roi.get_shape())
                else:
                    request[restrictive_mask_type].roi =\
                     request[restrictive_mask_type].roi.union(request_roi)

                # this node will provide this volume type
                del request[volume_type]
            else:
                # do nothing if no blobs of this type were requested
                logger.warning('%s output volume type for %s never requested. \
                    Deleting entry...'%(settings['output_volume_type'], blob_name))
                del self.blob_settings[blob_name]


    def process(self, batch, request):

        # check volumes and gather all IDs and synapse IDs
        all_points = {}
        all_synapse_ids = {}

        for blob_name, settings in self.blob_settings.items():
            # Unpack settings
            point_type = settings['point_type']
            restrictive_mask_type = settings['restrictive_mask_type']


            # Make sure both the necesary point types and volumes are present
            assert point_type in batch.points, "Upstream does not provide required point type\
            : %s"%point_type

            assert restrictive_mask_type in batch.volumes, "Upstream does not provide required \
            volume type: %s"%restrictive_mask_type

            # Get point data
            points = batch.points[point_type]

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

            all_synapse_ids[point_type] = synapse_ids
            all_points[point_type] = points

        for blob_name, settings in self.blob_settings.items():

            # Unpack settings
            point_type = settings['point_type']
            volume_type = settings['output_volume_type']
            voxel_size = settings['output_voxel_size']
            restrictive_mask_type = settings['restrictive_mask_type']
            restrictive_mask = batch.volumes[restrictive_mask_type].crop(request[volume_type].roi)

            id_mapper = settings['id_mapper']
            dtype = settings['output_volume_dtype']

            if id_mapper is not None:
                id_mapper.make_map(all_points)

            # Initialize output volume
            shape_volume = np.asarray(request[volume_type].roi.get_shape())/voxel_size
            blob_map = np.zeros(shape_volume, dtype=dtype)

            # Get point data
            points = batch.points[point_type]


            offset = np.asarray(points.spec.roi.get_offset())
            for point_id, point_data in points.data.items():
                voxel_location = np.round(((point_data.location - offset)/(voxel_size))).astype('int32')

                synapse_id = point_data.synapse_id
                # if mapping exists, do it
                if id_mapper is not None:
                    synapse_id = id_mapper(synapse_id)

                settings['blob_placer'].place(blob_map, voxel_location,
                    synapse_id, restrictive_mask.data)


            # Provide volume
            batch.volumes[volume_type] = Volume(blob_map, spec=request[volume_type].copy())
            batch.volumes[volume_type].spec.dtype = dtype

            # add id_mapping to attributes
            if id_mapper is not None:
                id_map_list = np.array(list(id_mapper.get_map().items()))
                batch.volumes[volume_type].attrs['id_mapping'] = id_map_list

            batch.volumes[volume_type].attrs['point_ids'] = points.data.keys()
            batch.volumes[volume_type].attrs['synapse_ids'] = all_synapse_ids[point_type]

            # Crop all other requests
        for volume_type, volume in request.volume_specs.items():
            batch.volumes[volume_type] = batch.volumes[volume_type].crop(volume.roi)

        for points_type, points in request.points_specs.items():
            batch.points[points_type] = batch.points[points_type].spec.roi = points.roi

class BlobPlacer:
    ''' Places synapse volume blobs from location data.
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

        self.sphere_voxel_volume = np.sum(self.sphere_map, axis=(0, 1, 2))

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
