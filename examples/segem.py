from gunpowder import *
import math
import time
import random
import malis

def train():

    set_verbose()

    random.seed(42)

    affinity_neighborhood = malis.mknhood3d()
    print(affinity_neighborhood)

    # simulate many sources (here we point to the same file always)
    sources = tuple(
        Hdf5Source(
                '100.hdf',
                raw_dataset='volumes/raw',
                gt_dataset='volumes/labels/neuron_ids',
                gt_mask_dataset='volumes/labels/mask') +
        Normalize() +
        Padding([100,100,100]) +
        RandomLocation()
        for i in range(10)
    )

    # create a batch provider by concatenation of filters
    batch_provider = (
            sources +
            RandomProvider() +
            ElasticAugmentation([40,40,40], [0,1,1], [0,math.pi/2.0]) +
            # For SegEM, we have to mask-reject after the elastic augmentation, 
            # otherwise we get stuck in requests that can't be fullfilled: If 
            # rotation is pi/2, for example, there is no way to have more than 
            # 50% GT in the requested output ROI upstream. The requested output 
            # ROI would be much larger than the GT ROI provided by a source.
            Reject() +
            SimpleAugment(transpose_only_xy=True) +
            IntensityAugment(0.9, 1.1, -0.1, 0.1) +
            AddGtAffinities(affinity_neighborhood) +
            # Needed for real training:
            # IntensityScaleShift(2, -1) +
            PreCache(
                    lambda : BatchSpec(
                            (144,188,188),
                            (96,100,100),
                            with_gt=True,
                            with_gt_mask=False,
                            with_gt_affinities=True),
                    cache_size=10,
                    num_workers=20) +
            Snapshot(every=1) +
            # Needed for real training (get it from gunpowder.caffe):
            # Train() +
            PrintProfilingStats()
    )

    n = 1000
    print("Training for " + str(n) + " iterations")
    with build(batch_provider) as b:
        for i in range(n):
            b.request_batch(None)
    print("Training finished")

if __name__ == "__main__":
    train()
