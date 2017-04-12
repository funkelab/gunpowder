import gunpowder
import math
import time

source1 = gunpowder.Hdf5Source('test2.hdf', raw_dataset='volumes/raw', gt_dataset='volumes/labels/neuron_ids', gt_mask_dataset='volumes/labels/mask')
source2 = gunpowder.Hdf5Source('test2.hdf', raw_dataset='volumes/raw', gt_dataset='volumes/labels/neuron_ids', gt_mask_dataset='volumes/labels/mask')
choose = gunpowder.RandomProvider()
snapshot_original = gunpowder.Snapshot(every=1, output_dir='snapshots_original')
random1 = gunpowder.RandomLocation()
random2 = gunpowder.RandomLocation()
reject = gunpowder.Reject()
jitter = gunpowder.ElasticAugmentation([1,20,20], [0,2,2], [0,math.pi/2.0])
simple_augment = gunpowder.SimpleAugment(transpose_only_xy=True)
grow_boundary = gunpowder.GrowBoundary(steps=3, only_xy=True)
defect_augment = gunpowder.DefectAugment(prob_missing=0.1, prob_low_contrast=0.1, contrast_scale=0.1)
snapshot_final = gunpowder.Snapshot(every=1)
# SegEM sized batch
# precache = gunpowder.PreCache(lambda : gunpowder.BatchSpec((144,188,188), with_gt=True, with_gt_mask=True), cache_size=20, num_workers=10)
precache = gunpowder.PreCache(lambda : gunpowder.BatchSpec((100,50,50), with_gt=True, with_gt_mask=True), cache_size=20, num_workers=10)
# precache = gunpowder.PreCache(lambda : gunpowder.BatchSpec((10,10,10), with_gt=True, with_gt_mask=True), cache_size=5, num_workers=2)

choose.add_upstream_provider(random1).add_upstream_provider(source1)
choose.add_upstream_provider(random2).add_upstream_provider(source2)

snapshot_final.\
        add_upstream_provider(precache).\
        add_upstream_provider(defect_augment).\
        add_upstream_provider(grow_boundary).\
        add_upstream_provider(jitter).\
        add_upstream_provider(simple_augment).\
        add_upstream_provider(reject).\
        add_upstream_provider(snapshot_original).\
        add_upstream_provider(choose)

snapshot_final.initialize_all()

start = time.time()
num_batches = 10
for i in range(num_batches):
    batch = snapshot_final.request_batch(None)

print("Got %d batches in %fs"%(num_batches,time.time()-start))
