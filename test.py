import gunpowder
import math
import time

source1 = gunpowder.Hdf5Source('test2.hdf', raw_dataset='volumes/raw', gt_dataset='volumes/labels/neuron_ids', gt_mask_dataset='volumes/labels/mask')
source2 = gunpowder.Hdf5Source('test2.hdf', raw_dataset='volumes/raw', gt_dataset='volumes/labels/neuron_ids', gt_mask_dataset='volumes/labels/mask')
choose = gunpowder.RandomProvider()
random1 = gunpowder.RandomLocation()
random2 = gunpowder.RandomLocation()
reject = gunpowder.Reject()
jitter = gunpowder.ElasticAugmentation([1,20,20], [0,2,2], [math.pi/4.0,math.pi/4.0])
grow_boundary = gunpowder.GrowBoundary(steps=3, only_xy=True)
snapshot = gunpowder.Snapshot(every=1)
# SegEM sized batch
precache = gunpowder.PreCache(lambda : gunpowder.BatchSpec((144,188,188), with_gt=True, with_gt_mask=True), cache_size=20, num_workers=10)
# precache = gunpowder.PreCache(lambda : gunpowder.BatchSpec((20,100,100), with_gt=True, with_gt_mask=True), cache_size=20, num_workers=10)
# precache = gunpowder.PreCache(lambda : gunpowder.BatchSpec((10,10,10), with_gt=True, with_gt_mask=True), cache_size=5, num_workers=2)

choose.add_upstream_provider(random1).add_upstream_provider(source1)
choose.add_upstream_provider(random2).add_upstream_provider(source2)

snapshot.\
        add_upstream_provider(precache).\
        add_upstream_provider(grow_boundary).\
        add_upstream_provider(jitter).\
        add_upstream_provider(reject).\
        add_upstream_provider(choose)

snapshot.initialize_all()

start = time.time()
num_batches = 10
for i in range(num_batches):
    batch = snapshot.request_batch(None)

print("Got %d batches in %fs"%(num_batches,time.time()-start))
