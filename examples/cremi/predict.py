from gunpowder import *
import random
import numpy as np

class DummyPrediction(BatchFilter):

    def process(self, batch):

        print("adding dummy predictions for " + str(batch.spec.output_roi))
        batch.prediction = np.zeros((3,) + batch.spec.output_roi.get_shape(), dtype=np.float32)
        batch.prediction[0,:] = random.random()
        batch.prediction[1,:] = random.random()
        batch.prediction[2,:] = random.random()
        batch.spec.with_prediction = True

def predict():

    chunk_size = Coordinate((200,200,200))

    pipeline = (
        Hdf5Source(
                'sample_A_20160501.hdf',
                raw_dataset='volumes/raw',
                gt_dataset='volumes/labels/neuron_ids') +
        Padding() +
        Normalize() +
        DummyPrediction() +
        # chunk into sizes that prediction can handle at once
        Chunk(
                BatchSpec(
                        chunk_size,
                        chunk_size - (5,5,5)
                )
        ) +
        Snapshot(every=1)
    )

    # request a "batch" of the size of the whole dataset
    with build(pipeline) as p:
        spec = p.get_spec()
        p.request_batch(
                BatchSpec(
                        spec.roi.get_shape(),
                        spec.roi.get_shape() - (5,5,5)
                )
        )

if __name__ == "__main__":
    predict()
