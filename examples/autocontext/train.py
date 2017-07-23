from __future__ import print_function

import math
import numpy as np
import random

from gunpowder import *

class DummyTrain(BatchFilter):
    def process(self, batch, request):
        pass

class AddDummyPredictions(BatchFilter):

    def process(self, batch, request):
        batch.volumes[VolumeTypes.PRED_AFFINITIES] = np.ones(
                request.volumes[VolumeTypes.PRED_AFFINITIES], dtype=np.float32
        )*0.5

def train():

    random.seed(42)
    set_verbose()

    affinity_neighborhood = np.array([[-1,0,0],[0,-1,0],[0,0,-1]])

    solver_parameters = SolverParameters()
    solver_parameters.train_net = 'net.prototxt'
    solver_parameters.base_lr = 1e-4
    solver_parameters.momentum = 0.95
    solver_parameters.momentum2 = 0.999
    solver_parameters.delta = 1e-8
    solver_parameters.weight_decay = 0.000005
    solver_parameters.lr_policy = 'inv'
    solver_parameters.gamma = 0.0001
    solver_parameters.power = 0.75
    solver_parameters.snapshot = 2000
    solver_parameters.snapshot_prefix = 'net'
    solver_parameters.type = 'Adam'
    solver_parameters.resume_from = None
    solver_parameters.train_state.add_stage('euclid')

    request = BatchRequest()
    request.add_volume_request(VolumeTypes.RAW, (84,268,268))
    request.add_volume_request(VolumeTypes.GT_LABELS, (56,56,56))
    request.add_volume_request(VolumeTypes.GT_IGNORE, (56,56,56))
    request.add_volume_request(VolumeTypes.GT_AFFINITIES, (56,56,56))
    request.add_volume_request(VolumeTypes.PRED_AFFINITIES, (84,268,268))

    data_sources = tuple(
        Hdf5Source(
            sample,
            raw_dataset='volumes/raw',
            gt_dataset='volumes/labels/neuron_ids',
        ) +
        Normalize() +
        RandomLocation()
        for sample in ['sample_A_20160501.hdf','sample_B_20160501.hdf','sample_C_20160501.hdf']
    )

    # use the same network for training and prediction to share weights

    # TODO: implement same_process
    train = Train(solver_parameters, same_process=True)
    # TODO: implement get_net()
    predict = Predict(train.get_net(), same_process=True)

    batch_provider_tree = (
        data_sources +
        RandomProvider() +
        ElasticAugment([4,40,40], [0,2,2], [0,math.pi/2.0]) +
        SimpleAugment(transpose_only_xy=True) +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        DefectAugment(prob_missing=0.1, prob_low_contrast=0.1, contrast_scale=0.1) +
        GrowBoundary(steps=3, only_xy=True) +
        AddGtAffinities(affinity_neighborhood) +
        IntensityScaleShift(2.0, -1.0) +
        ZeroOutConstSections() +
        SplitAndRenumberSegmentationLabels() +
        PreCache(
            cache_size=10,
            num_workers=5) +
        AddDummyPredictions() +
        # TODO: Predict needs to change size of RAW, and add PRED_AFFINITIES if 
        # part of network input
        predict +
        train
        Snapshot(every=1, output_filename='{id}.hdf')
    )

    n = 10
    print("Training for", n, "iterations")

    with build(batch_provider_tree) as minibatch_maker:
        for i in range(n):
            minibatch_maker.request_batch(request)

    print("Finished")


if __name__ == "__main__":
    train()
