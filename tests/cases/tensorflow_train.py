import numpy as np
from gunpowder import *
from gunpowder.tensorflow import Train
# from gunpowder.ext import tensorflow as tf
import tensorflow as tf
from .provider_test import ProviderTest

register_volume_type('A')
register_volume_type('B')
register_volume_type('C')
register_volume_type('GRADIENT_A')

class TestTensorflowTrainSource(BatchProvider):

    def setup(self):

        spec = VolumeSpec(
            roi=Roi((0, 0), (2, 2)),
            dtype=np.float32,
            interpolatable=True,
            voxel_size=(1, 1))
        self.provides(VolumeTypes.A, spec)
        self.provides(VolumeTypes.B, spec)

    def provide(self, request):

        batch = Batch()

        spec = self.spec[VolumeTypes.A]
        spec.roi = request[VolumeTypes.A].roi

        batch.volumes[VolumeTypes.A] = Volume(
            np.array([[0, 1], [2, 3]], dtype=np.float32),
            spec)

        spec = self.spec[VolumeTypes.B]
        spec.roi = request[VolumeTypes.B].roi

        batch.volumes[VolumeTypes.B] = Volume(
            np.array([[0, 1], [2, 3]], dtype=np.float32),
            spec)

        return batch

class TestTensorflowTrain(ProviderTest):

    def create_meta_graph(self):

        # create a tf graph
        a = tf.placeholder(tf.float32, shape=(2, 2))
        b = tf.placeholder(tf.float32, shape=(2, 2))
        v = tf.Variable(1, dtype=tf.float32)
        c = a*b*v

        # dummy "loss"
        loss = tf.norm(c)

        # dummy optimizer
        opt = tf.train.AdamOptimizer()
        optimizer = opt.minimize(loss)

        tf.train.export_meta_graph(filename='tf_graph.meta')

        return [x.name for x in [a, b, c, optimizer, loss]]

    def test_output(self):

        (a, b, c, optimizer, loss) = self.create_meta_graph()

        source = TestTensorflowTrainSource()
        train = Train(
            'tf_graph',
            optimizer=optimizer,
            loss=loss,
            inputs={VolumeTypes.A: a, VolumeTypes.B: b},
            outputs={VolumeTypes.C: c},
            gradients={VolumeTypes.GRADIENT_A: a})
        pipeline = source + train


        with build(pipeline):

            request = BatchRequest({
                VolumeTypes.A: VolumeSpec(roi=Roi((0, 0), (2, 2))),
                VolumeTypes.B: VolumeSpec(roi=Roi((0, 0), (2, 2))),
                VolumeTypes.C: VolumeSpec(roi=Roi((0, 0), (2, 2))),
                VolumeTypes.GRADIENT_A: VolumeSpec(roi=Roi((0, 0), (2, 2))),
            })

            batch = pipeline.request_batch(request)

            self.assertAlmostEqual(batch.loss, 9.8994951)

            gradient_a = batch.volumes[VolumeTypes.GRADIENT_A].data
            self.assertTrue(gradient_a[0, 0] < gradient_a[0, 1])
            self.assertTrue(gradient_a[0, 1] < gradient_a[1, 0])
            self.assertTrue(gradient_a[1, 0] < gradient_a[1, 1])

            for i in range(1000):
                loss1 = batch.loss
                batch = pipeline.request_batch(request)
                loss2 = batch.loss
                self.assertLess(loss2, loss1)
