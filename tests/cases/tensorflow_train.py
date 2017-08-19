from .provider_test import ProviderTest
from gunpowder import *
from gunpowder.tensorflow import Train
from gunpowder.ext import tensorflow as tf

register_volume_type(VolumeType('A', interpolate=True))
register_volume_type(VolumeType('B', interpolate=True))
register_volume_type(VolumeType('C', interpolate=True))
VolumeTypes.A.voxel_size = (1,1)
VolumeTypes.B.voxel_size = (1,1)
VolumeTypes.C.voxel_size = (1,1)

class TestTensorflowTrainSource(BatchProvider):

    def get_spec(self):

        spec = ProviderSpec()
        spec.volumes[VolumeTypes.A] = Roi((0,0), (2,2))
        spec.volumes[VolumeTypes.B] = Roi((0,0), (2,2))

        return spec

    def provide(self, request):

        batch = Batch()

        batch.volumes[VolumeTypes.A] = Volume(
                    [[0,1],[2,3]],
                    request.volumes[VolumeTypes.A])
        batch.volumes[VolumeTypes.B] = Volume(
                    [[0,1],[2,3]],
                    request.volumes[VolumeTypes.B])

        return batch

class TestTensorflowTrain(ProviderTest):

    def test_output(self):

        # create a tf graph
        a = tf.placeholder(tf.float32, shape=(2,2))
        b = tf.placeholder(tf.float32, shape=(2,2))
        v = tf.Variable(1, dtype=tf.float32)
        c = a*b*v

        # dummy "loss"
        loss = tf.norm(c)

        # dummy optimizer
        opt = tf.train.AdamOptimizer()
        optimizer = opt.minimize(loss)

        source = TestTensorflowTrainSource()
        pipeline = source + Train(
                optimizer = optimizer,
                loss = loss,
                inputs = {VolumeTypes.A: a, VolumeTypes.B: b},
                outputs = {VolumeTypes.C: c},
                gradients = {})

        request = BatchRequest({
            VolumeTypes.A: source.get_spec().volumes[VolumeTypes.A],
            VolumeTypes.B: source.get_spec().volumes[VolumeTypes.B]
        })

        with build(pipeline):
            batch = pipeline.request_batch(request)

        self.assertAlmostEqual(batch.loss, 9.8994951)
