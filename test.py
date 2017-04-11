import gunpowder
import math
import time

source = gunpowder.Hdf5Source('test.hdf', raw_dataset='volumes/raw', gt_dataset='volumes/labels/neuron_ids')
random = gunpowder.RandomLocation()
jitter = gunpowder.ElasticAugmentation([20,20,20], [2,2,2], [math.pi/4.0,math.pi/4.0])
snapshot = gunpowder.Snapshot()

jitter.add_upstream_provider(source)
random.add_upstream_provider(jitter)
snapshot.add_upstream_provider(random)

snapshot.initialize_all()

batch_spec = gunpowder.BatchSpec((144,188,188))

start = time.time()
batch = snapshot.request_batch(batch_spec)
print("Got batch in %fs"%(time.time()-start))
