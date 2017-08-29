from gunpowder import *
from gunpowder.caffe import *

def predict():

    # define which model and weights to use
    iteration = 200000
    prototxt = 'net.prototxt'
    weights  = 'net_iter_%d.caffemodel'%iteration

    # define a batch request
    request       = BatchRequest()
    roi_synapses  = Roi(offset=(9500, 2500, 8000), shape=(500, 500, 1500))
    shape_input     = roi_synapses.get_shape()
    shape_outputs = roi_synapses.get_shape()
    request.add_volume_request(VolumeTypes.RAW, shape_input)
    request.add_points_request(PointsTypes.PRESYN, shape_outputs)
    request.add_volume_request(VolumeTypes.GT_BM_PRESYN, shape_outputs)
    request.add_volume_request(VolumeTypes.GT_MASK_EXCLUSIVEZONE_PRESYN, shape_outputs)
    request.add_volume_request(VolumeTypes.PRED_BM_PRESYN, shape_outputs)

    # shift batch request roi to desired offset
    request_offset = roi_synapses.get_offset()
    for request_type in [request.volumes, request.points]:
        for type in request_type:
            request_type[type] += request_offset

    # create template for chunk node matching your network's architecture
    chunk_spec_template   = BatchRequest()
    shape_input_template  = [132, 132, 132]
    shape_output_template = [44, 44, 44]
    chunk_spec_template.add_volume_request(VolumeTypes.RAW, shape_input_template)
    chunk_spec_template.add_points_request(PointsTypes.PRESYN, shape_output_template)
    chunk_spec_template.add_volume_request(VolumeTypes.GT_BM_PRESYN, shape_output_template)
    chunk_spec_template.add_volume_request(VolumeTypes.GT_MASK_EXCLUSIVEZONE_PRESYN, shape_output_template)
    chunk_spec_template.add_volume_request(VolumeTypes.PRED_BM_PRESYN, shape_output_template)

    # define networks input, output names and VolumeTypes
    input_names_to_types  = {'data': VolumeTypes.RAW}
    output_names_to_types = {'bm_presyn_pred': VolumeTypes.PRED_BM_PRESYN}

    # set for padding synapse points
    padding_syn_points = (44, 44, 44)

    # define where to find data
    data_source = (DvidSource(
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
                                                       PointsTypes.PRESYN: roi_synapses.grow((0, 0, 0),
                                                                                             padding_syn_points),
                                                     },
                                resolution = (8,8,8)
                              ) +
                   Pad(
                       {VolumeTypes.RAW: (88, 88, 88)},
                       {VolumeTypes.RAW: 255}
                      ) +
                    Normalize())


    # define pipeline to process batches
    batch_provider_tree = (
            data_source +
            AddGtBinaryMapOfPoints({PointsTypes.PRESYN: VolumeTypes.GT_BM_PRESYN}) +
            AddGtMaskExclusiveZone()+
            IntensityScaleShift(2, -1) +
            ZeroOutConstSections() +
            Predict(prototxt, weights, input_names_to_types, output_names_to_types, use_gpu=0) +
            Chunk(chunk_spec_template) +
            Snapshot(every=1, output_filename='predictions.hdf')
            )

    # request a "batch" of the size of the whole dataset
    print("Inference started")
    with build(batch_provider_tree) as minibatch_maker:
        minibatch_maker.request_batch(request)
    print("Finished")


if __name__ == "__main__":
    predict()
