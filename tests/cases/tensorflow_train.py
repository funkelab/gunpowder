import numpy as np
import glob
import os
from gunpowder import *
from gunpowder.tensorflow import Train, Predict
# from gunpowder.ext import tensorflow as tf
import tensorflow as tf
from .provider_test import ProviderTest

register_array_type('A')
register_array_type('B')
register_array_type('C')
register_array_type('GRADIENT_A')

class TestTensorflowTrainSource(BatchProvider):

    def setup(self):

        spec = ArraySpec(
            roi=Roi((0, 0), (2, 2)),
            dtype=np.float32,
            interpolatable=True,
            voxel_size=(1, 1))
        self.provides(ArrayTypes.A, spec)
        self.provides(ArrayTypes.B, spec)

    def provide(self, request):

        batch = Batch()

        spec = self.spec[ArrayTypes.A]
        spec.roi = request[ArrayTypes.A].roi

        batch.arrays[ArrayTypes.A] = Array(
            np.array([[0, 1], [2, 3]], dtype=np.float32),
            spec)

        spec = self.spec[ArrayTypes.B]
        spec.roi = request[ArrayTypes.B].roi

        batch.arrays[ArrayTypes.B] = Array(
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

        set_verbose(False)

        # start clean
        for filename in glob.glob('tf_graph.*'):
            os.remove(filename)
        for filename in glob.glob('tf_graph_checkpoint_*'):
            os.remove(filename)
        try:
            os.remove('checkpoint')
        except:
            pass

        # create model meta graph file and get input/output names
        (a, b, c, optimizer, loss) = self.create_meta_graph()

        source = TestTensorflowTrainSource()
        train = Train(
            'tf_graph',
            optimizer=optimizer,
            loss=loss,
            inputs={a: ArrayTypes.A, b: ArrayTypes.B},
            outputs={c: ArrayTypes.C},
            gradients={a: ArrayTypes.GRADIENT_A},
            save_every=100)
        pipeline = source + train

        request = BatchRequest({
            ArrayTypes.A: ArraySpec(roi=Roi((0, 0), (2, 2))),
            ArrayTypes.B: ArraySpec(roi=Roi((0, 0), (2, 2))),
            ArrayTypes.C: ArraySpec(roi=Roi((0, 0), (2, 2))),
            ArrayTypes.GRADIENT_A: ArraySpec(roi=Roi((0, 0), (2, 2))),
        })

        # train for a couple of iterations
        with build(pipeline):

            batch = pipeline.request_batch(request)

            self.assertAlmostEqual(batch.loss, 9.8994951)

            gradient_a = batch.arrays[ArrayTypes.GRADIENT_A].data
            self.assertTrue(gradient_a[0, 0] < gradient_a[0, 1])
            self.assertTrue(gradient_a[0, 1] < gradient_a[1, 0])
            self.assertTrue(gradient_a[1, 0] < gradient_a[1, 1])

            for i in range(200-1):
                loss1 = batch.loss
                batch = pipeline.request_batch(request)
                loss2 = batch.loss
                self.assertLess(loss2, loss1)

        # resume training
        with build(pipeline):

            for i in range(100):
                loss1 = batch.loss
                batch = pipeline.request_batch(request)
                loss2 = batch.loss
                self.assertLess(loss2, loss1)

        # predict
        source = TestTensorflowTrainSource()
        predict = Predict(
            'tf_graph_checkpoint_300',
            inputs={a: ArrayTypes.A, b: ArrayTypes.B},
            outputs={c: ArrayTypes.C})
        pipeline = source + predict

        request = BatchRequest({
            ArrayTypes.A: ArraySpec(roi=Roi((0, 0), (2, 2))),
            ArrayTypes.B: ArraySpec(roi=Roi((0, 0), (2, 2))),
            ArrayTypes.C: ArraySpec(roi=Roi((0, 0), (2, 2))),
        })

        with build(pipeline):

            prev_c = None

            for i in range(100):
                batch = pipeline.request_batch(request)
                c = batch.arrays[ArrayTypes.C].data

                if prev_c is not None:
                    self.assertTrue(np.equal(c, prev_c))
                    prev_c = c
