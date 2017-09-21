import logging
import numpy as np
import itertools

from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class AddBlobsFromPoints(BatchFilter):

    def __init__(self, blob_settings):

        self.blob_settings = blob_settings

        for point_type, settings in self.blob_settings.items():
            blob_settings[point_type]['blob_placer'] = BlobPlacer(
                radius = settings['radius'], 
                resolution = settings['output_voxel_size'], 
                mask = settings['restrictive_mask_type']
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

                # If point is not already requested, add to request
                if point_type not in request.points_specs:
                    request.add(point_type, request[volume_type].roi.get_shape())

                # this node will provide this volume type
                del request[volume_type]
            else:
                # do nothing if no blobs of this type were requested
                logger.warning('{} output volume type for {} never requested. \
                    Deleting entry...'.format(settings['output_volume_type'], blob_name))
                del self.blob_settings[blob_name]


    def process(self, batch, request):

        for blob_name, settings in self.blob_settings.items():
            point_type = settings['point_type']
            volume_type = settings['output_volume_type']
            voxel_size = settings['output_voxel_size']
            restrictive_mask_type = settings['restrictive_mask_type']

            assert point_type in batch.points, "Upstream does not provide required point type\
            : {}".format(point_type)

            assert restrictive_mask_type in batch.volumes, "Upstream does not provide required \
            volume type: {}".format(restrictive_mask_type)

            points = batch.points[point_type]
            IDs = points.data.keys()


            shape_volume  = np.asarray(request[volume_type].roi.get_shape())/voxel_size
            voxel_offset = np.asarray(request[volume_type].roi.get_offset())/voxel_size

            output_volume = np.zeros(shape_volume)

            for ID in IDs:
                pass

class BlobPlacer:
    def __init__(self, radius, resolution, mask):
        ''' Places synapse volume blobs from location data.
        Args:
            radius: int - that desired radius of synaptic blobs
            resolution: array, list, tuple - voxel size in physical 
            mask:   3D np array - when placing a blob, will sample mask at 
        center location and only place blob in interection where mask has 
        the same ID. Usually used to restrict synaptic blobs inside their 
        respective cells (using segmentation)
        '''
        self.resolution = resolution
        if type(self.resolution) is list or type(self.resolution) is tuple:
            self.resolution = np.asarray(self.resolution)
        self.r = (radius/self.resolution)
        self.sphere_map = np.zeros(self.r*2)
        self.center = (np.asarray(self.sphere_map.shape))/2

        ranges = [range(0,self.r[0]*2),range(0,self.r[1]*2),range(0,self.r[2]*2)]
        for index in np.asarray(list(itertools.product(*ranges))):
            # if distance less than r, place a 1
            if np.linalg.norm((self.center-index)*self.resolution) <= radius:
                self.sphere_map[tuple(index)] = 1
        
        self.mask = mask
        self.sphere_voxel_volume = np.sum(self.sphere_map, axis=(0,1,2))
    
    def place(self, matrix, offset, marker):    
        ''' Places synapse
        Args:
            matrix: 4D np array - 1st dim are for layers to avoid overlap 
            (3 should be more than enough) 
            offset: np array - location where to place synaptic blob within given matrix
            marker: int - the ID used to mark this paricular synapse in the matrix
        '''
        # Calculate cube circumscribing the sphere to place
        start = offset - self.r
        end = offset + self.r
        
        # check if sphere fits in matrix
        if np.all(start >= 0) and np.all(np.asarray(matrix.shape) - end >= 0):    
            
            # calculate actual synapse shape from intersection between sphere and restrictive mask
            restricting_label = self.mask[offset[0],offset[1],offset[2]]
            restricting_mask = self.mask[start[0]:end[0],start[1]:end[1],start[2]:end[2]] == restricting_label
            shape = (self.sphere_map*restricting_mask)
            
            # place shape in chosen layer
            matrix[start[0]:end[0],start[1]:end[1],start[2]:end[2]] += shape*marker
            return matrix, True
        else:
            #print('Location'.format(settings['output_volume_type'], blob_name))
            print('Location {} out of bounds'.format(offset))
            return matrix, False
