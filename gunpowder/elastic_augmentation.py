import augment
import numpy as np
import itertools
import random
from batch_filter import BatchFilter

class ElasticAugmentation(BatchFilter):
    '''Elasticly deform a batch. Requests larger batches upstream to avoid data 
    loss due to rotation and jitter.'''

    def __init__(self, control_point_spacing, jitter_sigma, rotation_interval):

        self.control_point_spacing = control_point_spacing
        self.jitter_sigma = jitter_sigma
        self.rotation_start = rotation_interval[0]
        self.rotation_max_amount = rotation_interval[1] - rotation_interval[0]

    def prepare(self, batch_spec):

        output_shape = batch_spec.shape
        dims = len(output_shape)

        # create a transformation
        rotation = random.random()*self.rotation_max_amount + self.rotation_start
        self.transformation = augment.create_identity_transformation(output_shape)
        self.transformation += augment.create_elastic_transformation(
                output_shape,
                self.control_point_spacing,
                self.jitter_sigma)
        self.transformation += augment.create_rotation_transformation(
                output_shape,
                rotation)

        # get bounding box of needed data (plus save padding of 1)
        bb_min = tuple(int(self.transformation[d].min()) - 1 for d in range(dims))
        bb_max = tuple(int(self.transformation[d].max()) + 2 for d in range(dims))

        # request batch sufficiently large
        batch_spec.shape = tuple(bb_max[d] - bb_min[d] for d in range(dims))

        # if the offset was set, we need to shift it
        if batch_spec.offset is not None:
            batch_spec.offset = tuple(
                    batch_spec.offset[d] - (batch_spec.shape[d] - output_shape[d])/2
                    for d in range(dims)
            )

        # shift transformation into request batch
        for d in range(dims):
            self.transformation[d] -= bb_min[d]

        print("requested batch of " + str(output_shape))
        print("Need data in " + str(bb_min) + " -- " + str(bb_max))

    def process(self, batch):

        batch.raw = augment.apply_transformation(batch.raw, self.transformation, interpolate=True)
        if batch.gt is not None:
            batch.gt = augment.apply_transformation(batch.gt, self.transformation, interpolate=False)
        if batch.gt_mask is not None:
            batch.gt_mask = augment.apply_transformation(batch.gt_mask, self.transformation, interpolate=False)
