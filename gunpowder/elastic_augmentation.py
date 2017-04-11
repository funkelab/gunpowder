import augment
import numpy as np
import itertools
import random
from batch_filter import BatchFilter

class ElasticAugmentation(BatchFilter):
    '''Elasticly deform a batch. Requests larger batches upstream to avoid data 
    loss due to rotation and jitter.'''

    def __init__(self, num_control_points, jitter_sigma, rotation_interval):

        super(ElasticAugmentation, self).__init__()

        self.num_control_points = num_control_points
        self.jitter_sigma = jitter_sigma
        self.rotation_start = rotation_interval[0]
        self.rotation_max_amount = rotation_interval[1] - rotation_interval[0]

    def prepare(self, batch_spec):

        shape = batch_spec.shape

        self.rotation = random.random()*self.rotation_max_amount + self.rotation_start
        self.__update_batch_spec(self.rotation, batch_spec)

        self.transformation = augment.create_identity_transformation(shape)

        self.transformation += augment.create_elastic_transformation(
                shape,
                self.num_control_points,
                self.jitter_sigma)

        self.transformation += augment.create_rotation_transformation(
                shape,
                self.rotation)

    def process(self, batch):

        batch.raw = augment.apply_transformation(batch.raw, self.transformation)
        if batch.gt is not None:
            batch.gt = augment.apply_transformation(batch.gt, self.transformation)
        if batch.gt_mask is not None:
            batch.gt_mask = augment.apply_transformation(batch.gt_mask, self.transformation)

    def __update_batch_spec(self, rotation, batch_spec):

        # rotation padding
        dims = len(batch_spec.shape)
        keypoints = [
            tuple(batch_spec.shape[d]*select[d] for d in range(dims))
            for select in itertools.product(*[[0,1]]*dims)
        ]
        rotated = np.array(map(lambda p: augment.transform.rotate(p, -rotation), keypoints))
        mins = [ rotated[:,d].min() for d in range(dims) ]
        maxs = [ rotated[:,d].max() for d in range(dims) ]

        request_shape = list(maxs[d] - mins[d] for d in range(dims))

        # pad by 2*jitter sigma in each dimension
        for d in range(dims):
            request_shape[d] += 2*self.jitter_sigma[d]

        batch_spec.shape = tuple(request_shape)
        return batch_spec
