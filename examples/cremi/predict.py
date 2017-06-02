import os

from gunpowder import *
from gunpowder.caffe import *

def predict():

    iteration = 10000
    prototxt = 'net.prototxt'
    weights  = 'net_iter_%d.caffemodel'%iteration

    input_size = Coordinate((84,268,268))
    output_size = Coordinate((56,56,56))

    pipeline = (
            Hdf5Source(
                    'sample_A_20160501.hdf',
                    raw_dataset='volumes/raw') +
            Normalize() +
            Pad() +
            IntensityScaleShift(2, -1) +
            ZeroOutConstSections() +
            Predict(prototxt, weights, use_gpu=0) +
            Snapshot(
                    every=1,
                    output_dir=os.path.join('chunks', '%d'%iteration),
                    output_filename='chunk.hdf'
            ) +
            PrintProfilingStats() +
            Chunk(
                    BatchSpec(
                            input_size,
                            output_size
                    )
            ) +
            Snapshot(
                    every=1,
                    output_dir=os.path.join('processed', '%d'%iteration),
                    output_filename='sample_A_20160501.hdf'
            )
    )

    # request a "batch" of the size of the whole dataset
    with build(pipeline) as p:
        shape = p.get_spec().roi.get_shape()
        p.request_batch(
                BatchSpec(
                        shape,
                        shape - (input_size-output_size)
                )
        )

if __name__ == "__main__":
    predict()
