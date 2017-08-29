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
    loss due to rotation and jitter.

    Args:
        control_point_spacing (tuple of int): Distance between control points 
            for the elastic deformation, in voxels per dimension.

        jitter_sigma (tuple of float): Standard deviation of control point 
            jitter distribution, in voxels per dimension.

        rotation_interval (two floats): Interval to randomly sample rotation 
            angles from (0,2PI).

        prob_slip (float): Probability of a section to "slip", i.e., be 
            independently moved in x-y.

        prob_shift (float): Probability of a section and all following sections 
            to move in x-y.

        max_misalign (int): Maximal voxels to shift in x and y. Samples will be 
            drawn uniformly.

        subsample (int): Instead of creating an elastic transformation on the 
            full resolution, create one subsampled by the given factor, and 
            linearly interpolate to obtain the full resolution transformation. 
            This can significantly speed up this node, at the expense of having 
            visible piecewise linear deformations for large factors. Usually, a 
            factor of 4 can savely by used without noticable changes. However, 
            the default is 1 (i.e., no subsampling).
    '''

    def __init__(
            self,
            control_point_spacing,
            jitter_sigma,
            rotation_interval,
            prob_slip=0,
            prob_shift=0,
            max_misalign=0,
            subsample=1):

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

        self.voxel_size = None
        prev_volume_type = None
        for volume_type in request.volume_specs.keys():
            if self.voxel_size is None:
                self.voxel_size = self.spec[volume_type].voxel_size
            else:
                assert self.voxel_size == self.spec[volume_type].voxel_size, \
                        "ElasticAugment can only be used with volumes of same voxel sizes, " \
                        "but %s has %s, and %s has %s."%(
                                volume_type, self.spec[volume_type].voxel_size,
                                prev_volume_type, self.spec[prev_volume_type].voxel_size)
            prev_volume_type = volume_type

        # get total roi in voxels
        total_roi /= self.voxel_size

        # create a transformation for the total ROI
        self.total_transformation = augment.create_identity_transformation(
                total_roi.get_shape(),
                subsample=self.subsample)
        if sum(self.jitter_sigma) > 0:
            self.total_transformation += augment.create_elastic_transformation(
                    total_roi.get_shape(),
                    self.control_point_spacing,
                    self.jitter_sigma,
                    subsample=self.subsample)
        rotation = random.random()*self.rotation_max_amount + self.rotation_start
        if rotation != 0:
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
        for identifier, spec in request.items():

            roi = spec.roi

            logger.debug("downstream request ROI for %s is %s" % (identifier, roi))

            # get roi in voxels
            roi /= self.voxel_size

            roi_in_total_roi = roi.shift(-total_roi.get_offset())

            transformation = np.copy(
                self.total_transformation[(slice(None),) + roi_in_total_roi.get_bounding_box()]
            )
            self.transformations[identifier] = transformation

            # update request ROI to get all voxels necessary to perfrom
            # transformation
            spec.roi = self.__recompute_roi(roi, transformation)*self.voxel_size

            logger.debug("upstream request roi for %s = %s" % (identifier, spec.roi))


    def process(self, batch, request):

        for (volume_type, volume) in batch.volumes.items():

            # apply transformation
            volume.data = augment.apply_transformation(
                    volume.data,
                    self.transformations[volume_type],
                    interpolate=self.spec[volume_type].interpolatable)

            # restore original ROIs
            volume.spec.roi = request[volume_type].roi

        for (points_type, points) in batch.points.items():
            # create map/volume from points and apply tranformation to corresponding map, reconvert map back to points
            # TODO: How to avoid having to allocate a new volume each time (rather reuse,
            # but difficult since shape is alternating)
            trial_nr, max_trials, all_points_mapped = 0, 5, False
            shape_map = points.spec.roi.get_shape()/self.voxel_size
            id_map_volume = np.zeros(shape_map, dtype=np.int32)

            # Get all points located in current batch and shift it based on absolute offset. Assign new ids to point
            # ids to have positive consecutive numbers and to account for having multiple points with same location.
            ids_not_mapped = []
            offset_volume = points.spec.roi.get_offset()
            new_pointid_to_ori_pointid = {} # maps new id to original id(s)
            new_point_id = 1
            relabeled_points_dic = {} # new dictionary including only those points that are relevant for that batch
            location_to_pointid_dic = {}
            for point_id, point in points.data.items():
                location = point.location
                if points.spec.roi.contains(Coordinate(location)):
                    location = location - np.asarray(offset_volume)
                    if tuple(location) in location_to_pointid_dic.keys():
                        id_with_same_location = location_to_pointid_dic[tuple(location)]
                        new_pointid_to_ori_pointid[id_with_same_location].append(point_id)
                        logging.debug("point with id %i has same location as other points eg. point id "
                                      "%i" %(point_id, new_pointid_to_ori_pointid[id_with_same_location][0]))
                    else:
                        new_pointid_to_ori_pointid[new_point_id] = [point_id]
                        ids_not_mapped.append(new_point_id)
                        relabeled_points_dic[new_point_id] = location
                        location_to_pointid_dic[tuple(location)] = new_point_id
                        new_point_id += 1
                else:
                    del points.data[point_id]

            while not all_points_mapped:
                marker_size = trial_nr # for each trial, the region of the point in the map is increased to increase
                # the likelihood that the point still exists after transformation
                id_map = self.__from_points_to_idmap(relabeled_points_dic, id_map_volume,
                                                     ids_not_mapped, marker_size=marker_size)
                id_map = augment.apply_transformation(
                    id_map,
                    self.transformations[points_type],
                    interpolate=False)
                ids_in_map = list(np.unique(id_map))
                ids_in_map.remove(0)
                ids_not_mapped =set(ids_not_mapped) - set(ids_in_map)
                self.__from_idmap_to_points(relabeled_points_dic, id_map, ids_in_map)
                trial_nr += 1
                if len(ids_not_mapped) == 0:
                    all_points_mapped = True
                elif trial_nr == max_trials:
                    for point_id in ids_not_mapped:
                        logger.debug("point %i with location %s was removed during "
                                     "elastic augmentation." % (new_pointid_to_ori_pointid[point_id][0],
                                                                relabeled_points_dic[point_id]))
                        for ori_point_id in new_pointid_to_ori_pointid[point_id]:
                            del points.data[ori_point_id]
                            if point_id in relabeled_points_dic:
                                del relabeled_points_dic[point_id]

                    all_points_mapped = True
                else:
                    id_map_volume.fill(0)

            # restore original ROIs
            points.spec.roi = request[points_type].roi

            # assign new transformed location and map new ids back to original ids.
            for new_point_id, location in relabeled_points_dic.items():
                for ori_point_id in new_pointid_to_ori_pointid[new_point_id]:
                    points.data[ori_point_id].location = location + points.spec.roi.get_offset() # shift back to original roi.



    def __from_idmap_to_points(self, points, id_map, ids_in_map):
        for point_id in ids_in_map:
            if point_id == 0:
                continue
            locations = zip(*np.where(id_map == point_id))
            # If there are more than one locations, heuristically grab the first one
            new_location = locations[0]
            points[point_id] = new_location*np.array(self.voxel_size)

    def __from_points_to_idmap(self, points, id_map_volume, ids_to_map, marker_size=0):
        # TODO: This is partially a duplicate of add_gt_binary_map_points:get_binary_map, refactor!
        relative_voxel_size = 1./(np.array(self.voxel_size)/np.min(np.array(self.voxel_size)))
        for point_id in ids_to_map:
            location = points[point_id].astype(np.int32)
            location /= self.voxel_size # convert locations in world units to voxel units
            if marker_size > 0:
                marker_locs = tuple(slice(max(0, location[dim] - int(np.floor(marker_size*relative_voxel_size[dim]))),
                                          min(id_map_volume.shape[dim] - 1,
                                              location[dim] + max(int(np.floor(marker_size*relative_voxel_size[dim])), 1)))
                                    for dim in range(len(location)))

            else:
                marker_locs = [[loc] for loc in location]

            id_map_volume[marker_locs] = point_id
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
