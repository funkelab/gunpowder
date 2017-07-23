from __future__ import print_function

from gunpowder import *
from gunpowder.nodes import *
from gunpowder.caffe import Train, net_input, net_output, net_gt, SolverParameters


def train(max_iteration):

    # define a batch request
    request      = BatchRequest()
    shape_input  = (132, 132, 132)
    shape_output = (44, 44, 44)
    request.add_volume_request(VolumeTypes.RAW, shape_input)
    request.add_points_request(PointsTypes.PRESYN, shape_output)
    request.add_volume_request(VolumeTypes.GT_BM_PRESYN, shape_output)
    request.add_volume_request(VolumeTypes.GT_MASK_EXCLUSIVEZONE_PRESYN, shape_output)

    # define networks input, output and groundtruth names and VolumeTypes
    net_inputs =  [
                    net_input(volume_name='data', volume_type=VolumeTypes.RAW)
                  ]

    net_outputs = [
                    net_output(volume_name='bm_presyn_pred', volume_type=VolumeTypes.PRED_BM_PRESYN,
                               gt_name='bm_presyn_label', loss_volume_type=VolumeTypes.LOSS_GRADIENT_PRESYN),
                  ]
    net_gts =     [
                    net_gt(volume_name='bm_presyn_label', volume_type=VolumeTypes.GT_BM_PRESYN,
                           scale_name='bm_presyn_scale', mask_volume_type=VolumeTypes.GT_MASK_EXCLUSIVEZONE_PRESYN),
                  ]

    # define solver parameters
    solver_parameters = SolverParameters()
    solver_parameters.train_net       = 'net.prototxt'
    solver_parameters.base_lr         = 1e-5
    solver_parameters.momentum        = 0.99
    solver_parameters.momentum2       = 0.999
    solver_parameters.delta           = 1e-8
    solver_parameters.weight_decay    = 0.000005
    solver_parameters.lr_policy       = 'inv'
    solver_parameters.gamma           = 0.0001
    solver_parameters.power           = 0.75
    solver_parameters.snapshot        = 10000
    solver_parameters.snapshot_prefix = 'net'
    solver_parameters.type            = 'Adam'
    solver_parameters.resume_from = None
    solver_parameters.train_state.add_stage('euclid')

    # define where to find data
    data_sources = (
                    DvidSource(
                                hostname = 'emdata2',
                                port     = 8000,
                                uuid     = 'cb7dc',
                                volume_array_names = {
                                                       VolumeTypes.RAW:       'grayscale',
                                                       VolumeTypes.GT_LABELS: 'labels'
                                                     },
                                points_array_names = {
                                                       PointsTypes.PRESYN: 'combined_synapses_08302016',
                                                     },
                                points_rois        = {
                                                       PointsTypes.PRESYN: Roi(offset=(9500, 2500, 8000),
                                                                               shape=(500, 500, 1500)),
                                                     },
                                resolution = (8,8,8)
                              ) +
                    RandomLocation(focus_points_type=PointsTypes.PRESYN) +
                    Normalize()
                    )

    # define pipeline to process batches
    batch_provider_tree = (
                            data_sources +
                            RandomProvider() +
                            AddGtBinaryMapOfPoints({PointsTypes.PRESYN:  VolumeTypes.GT_BM_PRESYN}) +
                            AddGtMaskExclusiveZone() +
                            SimpleAugment(transpose_only_xy=True) +
                            IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
                            IntensityScaleShift(2, -1) +
                            PreCache() +
                            Train(solver_parameters, net_inputs, net_outputs, net_gts, use_gpu=0) +
                            Snapshot(every=5000, output_filename='batch_{id}.hdf')
                          )

    print("Training for", max_iteration, "iterations")
    with build(batch_provider_tree) as minibatch_maker:
        for i in range(max_iteration):
            minibatch_maker.request_batch(request)
    print("Finished")


if __name__ == "__main__":
    set_verbose(False)
    train(max_iteration=200000)

