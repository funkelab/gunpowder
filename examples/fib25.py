from gunpowder import *

def train():

    set_verbose()

    sources = tuple(
            DvidSource(
                    'gs://flyem-public-connectome',
                    'CXtraining',
                    sample + '-grayscale',
                    sample + '-groundtruth',
                    ) +
            RandomLocation()
            for sample in ['pb1', 'pb2']
    )

    # create a batch provider by concatenation of filters
    batch_provider = (
            sources +
            RandomProvider() +
            PreCache(
                lambda : BatchSpec(
                    (256,256,256),
                    (256,256,256),
                    with_gt=True
            )) +
            Snapshot(every=1) +
            PrintProfilingStats()
    )

    n = 10
    print("Training for " + str(n) + " iterations")
    with build(batch_provider) as b:
        for i in range(n):
            b.request_batch(BatchSpec(
                    (256,256,256),
                    (256,256,256),
                    with_gt=True
            ))
    print("Training finished")

if __name__ == "__main__":
    train()
