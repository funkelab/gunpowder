import sys
from gunpowder import *
from gunpowder.caffe import *

def predict_affinities(gpu):

    set_verbose(False)

    # the network architecture
    prototxt = 'net.prototxt'

    # the learned weights (example at iteration 90000)
    weights  = 'net_iter_90000.caffemodel'

    # input and output sizes of the network (needed to formulate Chunk requests 
    # of the correct size later)
    input_size = Coordinate((196,)*3)
    output_size = Coordinate((92,)*3)

    # the size of the receptive field of the network
    context = (input_size - output_size)/2

    # a chunk request that matches the dimensions of the network, will be used 
    # to chunk the whole volume into batches of this size
    chunk_request = BatchRequest()
    chunk_request.add_volume_request(VolumeTypes.RAW, input_size)
    chunk_request.add_volume_request(VolumeTypes.PRED_AFFINITIES, output_size)

    # where to find the intensities
    source = Hdf5Source(
            'trvol-250-1.hdf',
            datasets = { VolumeTypes.RAW: 'volumes/raw'}
    )

    # the prediction pipeline:
    process_pipeline = (
            source +

            # ensure RAW is in float in [0,1]
            Normalize() +

            # zero-pad provided RAW to be able to draw batches close to the 
            # boundary of the available data
            Pad({ VolumeTypes.RAW: (100, 100, 100) }) +

            # ensure RAW is in [-1,1]
            IntensityScaleShift(2, -1) +

            # predict affinities
            Predict(prototxt, weights, use_gpu=gpu) +

            # add useful profiling stats to identify bottlenecks
            PrintProfilingStats() +

            # chunk the whole volume into chunk_request sized batches, this node 
            # requests several batches upstream, passes one downstream
            Chunk(chunk_request) +

            # save the prediction of the whole volume
            Snapshot(
                    every=1,
                    output_dir='processed',
                    output_filename='trvol-250-1.hdf'
            )
    )

    with build(process_pipeline) as p:

        # get the ROI of the whole RAW region from the source
        raw_roi = source.get_spec().volumes[VolumeTypes.RAW]

        # request affinity predictions for the whole RAW ROI
        whole_request = BatchRequest({
                VolumeTypes.RAW: raw_roi,
                VolumeTypes.PRED_AFFINITIES: raw_roi.grow(-context, -context)
            })

        print("Requesting " + str(whole_request) + " in chunks of " + str(chunk_request))

        p.request_batch(whole_request)

if __name__ == "__main__":
    gpu = int(sys.argv[1])
    predict_affinities(gpu)
