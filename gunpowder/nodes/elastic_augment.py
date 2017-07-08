import copy
import logging
import math
import numpy as np
import random

from .batch_filter import BatchFilter
from gunpowder.coordinate import Coordinate
from gunpowder.ext import augment
from gunpowder.roi import Roi
from gunpowder.volume import VolumeTypes

logger = logging.getLogger(__name__)

class ElasticAugment(BatchFilter):
    '''Elasticly deform a batch. Requests larger batches upstream to avoid data 
    loss due to rotation and jitter.'''

    def __init__(
            self,
            control_point_spacing,
            jitter_sigma,
            rotation_interval,
            prob_slip=0,
            prob_shift=0,
            max_misalign=0,
            subsample=1):
        '''Create an elastic deformation augmentation.

        Args:
            control_point_spacing: Distance between control points for the 
            elastic deformation, in voxels per dimension.

            jitter_sigma: Standard deviation of control point jitter 
            distribution, one value per dimension.

            rotation_interval: Interval to randomly sample rotation angles from 
            (0,2PI).

            prob_slip: Probability of a section to "slip", i.e., be 
            independently moved in x-y.

            prob_shift: Probability of a section and all following sections to 
            move in x-y.

            max_misalign: Maximal voxels to shift in x and y. Samples will be 
            drawn uniformly.

            subsample: Instead of creating an elastic transformation on the full 
            resolution, create one subsampled by the given factor, and linearly 
            interpolate to obtain the full resolution transformation. This can 
            significantly speed up this node, at the expense of having visible 
            piecewise linear deformations for large factors. Usually, a factor 
            of 4 can savely by used without noticable changes. However, the 
            default is 1 (i.e., no subsampling).
        '''

        self.control_point_spacing = control_point_spacing
        self.jitter_sigma = jitter_sigma
        self.rotation_start = rotation_interval[0]
        self.rotation_max_amount = rotation_interval[1] - rotation_interval[0]
        self.prob_slip = prob_slip
        self.prob_shift = prob_shift
        self.max_misalign = max_misalign
        self.subsample = subsample

    def prepare(self, request):

        total_roi = request.get_total_roi()
        logger.debug("total ROI is %s"%total_roi)
        dims = len(total_roi.get_shape())

        # create a transformation for the total ROI
        rotation = random.random()*self.rotation_max_amount + self.rotation_start
        self.total_transformation = augment.create_identity_transformation(
                total_roi.get_shape(),
                subsample=self.subsample)
        self.total_transformation += augment.create_elastic_transformation(
                total_roi.get_shape(),
                self.control_point_spacing,
                self.jitter_sigma,
                subsample=self.subsample)
        self.total_transformation += augment.create_rotation_transformation(
                total_roi.get_shape(),
                rotation,
                subsample=self.subsample)

        if self.subsample > 1:
            self.total_transformation = augment.upscale_transformation(
                    self.total_transformation,
                    total_roi.get_shape())

        if self.prob_slip + self.prob_shift > 0:
            self.__misalign()

        # crop the parts corresponding to the requested volume ROIs
        self.transformations = {}
        logger.debug("total ROI is %s"%total_roi)
        for collection_type in [request.volumes, request.points]:
            for type, roi in collection_type.items():
                logger.debug("downstream request ROI for %s is %s" % (type, roi))

                roi_in_total_roi = roi.shift(-total_roi.get_offset())

                transformation = np.copy(
                    self.total_transformation[(slice(None),) + roi_in_total_roi.get_bounding_box()]
                )
                self.transformations[type] = transformation

                # update request ROI to get all voxels necessary to perfrom
                # transformation
                roi = self.__recompute_roi(roi, transformation)
                collection_type[type] = roi

                logger.debug("upstream request roi for %s = %s" % (type, roi))


    def process(self, batch, request):

        for (volume_type, volume) in batch.volumes.items():

            # apply transformation
            volume.data = augment.apply_transformation(
                    volume.data,
                    self.transformations[volume_type],
                    interpolate=volume_type.interpolate)

            # restore original ROIs
            volume.roi = request.volumes[volume_type]

        for (points_type, points) in batch.points.items():
            # create map/volume from points and apply tranformation to corresponding map, reconvert map back to points
            # TODO: How to avoid having to allocate a new volume each time (rather reuse, but difficult since shape is alternating)
            id_map = self.__from_points_to_idmap(points)
            id_map = augment.apply_transformation(
                id_map,
                self.transformations[points_type],
                interpolate=False)
            self.__from_idmap_to_points(points, id_map)

            # restore original ROIs
            points.roi = request.points[points_type]

    def __from_idmap_to_points(self, points, id_map):
        ids_in_map = list(np.unique(id_map))
        for point_id in ids_in_map:
            if point_id == 0:
                continue
            locations = zip(*np.where(id_map == point_id))
            # If there are more than one locations, heuristically grab the first one
            new_location = locations[0]
            points.data[point_id].location = new_location

    def __from_points_to_idmap(self, points):
        # TODO: This is partially a duplicate of add_gt_binary_map_points:get_binary_map, refactor!
        shape_map = points.roi.get_shape()
        # TODO: Consider problem of negative or 0 id, maybe relabel point id list.
        id_map_volume = np.zeros(shape_map, dtype=np.int32)
        for point_id in points.data.keys():
            location = points.data[point_id].location
            if points.roi.contains(Coordinate(location)):
                # TODO: does not consider problem of having the same location for multiple point ids. Currently, the id is overwritten.
                id_map_volume[[[loc] for loc in location]] = point_id
        return id_map_volume

    def __recompute_roi(self, roi, transformation):

        dims = roi.dims()

        # get bounding box of needed data for transformation
        bb_min = Coordinate(int(math.floor(transformation[d].min())) for d in range(dims))
        bb_max = Coordinate(int(math.ceil(transformation[d].max())) + 1 for d in range(dims))

        # create roi sufficiently large to feed transformation
        source_roi = Roi(
                bb_min,
                bb_max - bb_min
        )

        # shift transformation, such that it can be applied on indices of source 
        # batch
        for d in range(dims):
            transformation[d] -= bb_min[d]

        return source_roi

    def __misalign(self):

        num_sections = self.total_transformation[0].shape[0]

        shifts = [Coordinate((0,0,0))]*num_sections
        for z in range(num_sections):

            r = random.random()

            if r <= self.prob_slip:

                shifts[z] = self.__random_offset()

            elif r <= self.prob_slip + self.prob_shift:

                offset = self.__random_offset()
                for zp in range(z, num_sections):
                    shifts[zp] += offset

        logger.debug("misaligning sections with " + str(shifts))

        dims = 3
        bb_min = tuple(int(math.floor(self.total_transformation[d].min())) for d in range(dims))
        bb_max = tuple(int(math.ceil(self.total_transformation[d].max())) + 1 for d in range(dims))
        logger.debug("min/max of transformation: " + str(bb_min) + "/" + str(bb_max))

        for z in range(num_sections):
            self.total_transformation[1][z,:,:] += shifts[z][1]
            self.total_transformation[2][z,:,:] += shifts[z][2]

        bb_min = tuple(int(math.floor(self.total_transformation[d].min())) for d in range(dims))
        bb_max = tuple(int(math.ceil(self.total_transformation[d].max())) + 1 for d in range(dims))
        logger.debug("min/max of transformation after misalignment: " + str(bb_min) + "/" + str(bb_max))

    def __random_offset(self):

        return Coordinate((0,) + tuple(self.max_misalign - random.randint(0, 2*int(self.max_misalign)) for d in range(2)))
