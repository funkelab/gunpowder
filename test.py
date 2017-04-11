import gunpowder
import math

source = gunpowder.Hdf5Source('test.hdf', raw_dataset='volumes/raw', gt_dataset='volumes/labels/neuron_ids')
random = gunpowder.RandomLocation()
jitter = gunpowder.ElasticAugmentation([2,2,2], [5,5,5], [0,math.pi/2.0])

random.add_upstream_provider(source)
jitter.add_upstream_provider(random)
jitter.initialize_all()

batch_spec = gunpowder.BatchSpec((10,10,10))
batch = jitter.request_batch(batch_spec)
