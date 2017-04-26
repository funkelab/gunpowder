from gunpowder import *

def train():

    set_verbose()

    # create a batch provider by concatenation of filters
    batch_provider = (
            DvidSource(
                    'gs://flyem-public-connectome',
                    'medulla-training',
                    'training2-grayscale',
                    'training2-groundtruth',
                    ) +
            RandomProvider() +
            Snapshot(every=1) +
            PrintProfilingStats()
    )

    n = 1
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
