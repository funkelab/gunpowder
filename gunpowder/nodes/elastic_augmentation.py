import copy
import logging
import math
import numpy as np
import random

from .batch_filter import BatchFilter
from gunpowder.ext import augment
from gunpowder.roi import Roi
from gunpowder.volume import VolumeType

logger = logging.getLogger(__name__)

class ElasticAugmentation(BatchFilter):
    '''Elasticly deform a batch. Requests larger batches upstream to avoid data 
    loss due to rotation and jitter.'''

    def __init__(self, control_point_spacing, jitter_sigma, rotation_interval):

        self.control_point_spacing = control_point_spacing
        self.jitter_sigma = jitter_sigma
        self.rotation_start = rotation_interval[0]
        self.rotation_max_amount = rotation_interval[1] - rotation_interval[0]

    def prepare(self, request):

        total_roi = request.get_total_roi()
        dims = len(total_roi.get_shape())

        # create a transformation for the total ROI
        rotation = random.random()*self.rotation_max_amount + self.rotation_start
        self.total_transformation = augment.create_identity_transformation(total_roi.get_shape())
        self.total_transformation += augment.create_elastic_transformation(
                total_roi.get_shape(),
                self.control_point_spacing,
                self.jitter_sigma)
        self.total_transformation += augment.create_rotation_transformation(
                total_roi.get_shape(),
                rotation)

        # crop the parts corresponding to the requested volume ROIs
        self.transformations = {}
        for (volume_type, roi) in request.volumes.items():

            logger.debug("downstream request roi for %s = %s"%(volume_type,roi))

            roi_in_total_roi = roi.shift(-total_roi.get_offset())

            transformation = np.copy(
                    self.total_transformation[(slice(None),)+roi_in_total_roi.get_bounding_box()]
            )
            self.transformations[volume_type] = transformation

            # update request ROI to get all voxels necessary to perfrom 
            # transformation
            roi = self.__recompute_roi(roi, transformation)
            request.volumes[volume_type] = roi

            logger.debug("upstream request roi for %s = %s"%(volume_type,roi))


    def process(self, batch, request):

        for (volume_type, volume) in batch.volumes.items():

            # apply transformation
            volume.data = augment.apply_transformation(
                    volume.data,
                    self.transformations[volume_type],
                    interpolate=volume.interpolate)

            # restore original ROIs
            volume.roi = request.volumes[volume_type]

    def __recompute_roi(self, roi, transformation):

        dims = roi.dims()

        # get bounding box of needed data for transformation
        bb_min = tuple(int(math.floor(transformation[d].min())) for d in range(dims))
        bb_max = tuple(int(math.ceil(transformation[d].max())) + 1 for d in range(dims))

        # create roi sufficiently large to feed transformation
        source_shape = tuple(bb_max[d] - bb_min[d] for d in range(dims))
        source_roi = Roi(
                roi.get_offset(),
                source_shape
        )

        # if the offset of roi was set, we need to shift it
        if roi.get_offset() is not None:
            shift = tuple(
                    (transformation.shape[d+1] - source_shape[d])/2
                    for d in range(dims)
            )
            source_roi = source_roi.shift(shift)

        # shift transformation, such that it can be applied on indices of source 
        # batch
        for d in range(dims):
            transformation[d] -= bb_min[d]

        return source_roi
