import copy
import logging
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

    def prepare(self, batch_spec):

        # remember ROIs to restore them later
        self.request_input_roi = copy.deepcopy(batch_spec.input_roi)
        self.request_output_roi = copy.deepcopy(batch_spec.output_roi)

        target_shape = batch_spec.input_roi.get_shape()
        dims = len(target_shape)

        # create a transformation for the input ROI
        rotation = random.random()*self.rotation_max_amount + self.rotation_start
        self.input_transformation = augment.create_identity_transformation(target_shape)
        self.input_transformation += augment.create_elastic_transformation(
                target_shape,
                self.control_point_spacing,
                self.jitter_sigma)
        self.input_transformation += augment.create_rotation_transformation(
                target_shape,
                rotation)

        # crop the part corresponding to the output ROI
        shift = tuple(-x for x in batch_spec.input_roi.get_offset())
        output_in_input_roi = batch_spec.output_roi.shift(shift)
        self.output_transformation = np.copy(self.input_transformation[(slice(None),)+output_in_input_roi.get_bounding_box()])

        batch_spec.input_roi = self.__recompute_roi(batch_spec.input_roi, self.input_transformation)
        batch_spec.output_roi = self.__recompute_roi(batch_spec.output_roi, self.output_transformation)

        logger.debug("downstream request shape = " + str(target_shape))
        logger.debug("upstream request shape = " + str(batch_spec.input_roi.get_shape()))

    def process(self, batch):

        for (volume_type, volume) in batch.volumes.items():
            volume.data = augment.apply_transformation(
                    volume.data,
                    self.input_transformation,
                    interpolate=volume.interpolate)

        batch.spec.input_roi = self.request_input_roi
        batch.spec.output_roi = self.request_output_roi

    def __recompute_roi(self, roi, transformation):

        dims = roi.dims()

        # get bounding box of needed data for transformation (plus save padding 
        # of 1)
        bb_min = tuple(int(transformation[d].min()) - 1 for d in range(dims))
        bb_max = tuple(int(transformation[d].max()) + 2 for d in range(dims))

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
