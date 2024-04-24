import multiprocessing

import numpy as np
import pytest

from gunpowder import (
    Array,
    ArrayKey,
    ArraySpec,
    Batch,
    BatchProvider,
    BatchRequest,
    Roi,
    build,
)
from gunpowder.ext import NoSuchModule, tensorflow
from gunpowder.tensorflow import Train


class ExampleTensorflowTrainSource(BatchProvider):
    def __init__(self, a_key, b_key):
        self.a_key = a_key
        self.b_key = b_key

    def setup(self):
        spec = ArraySpec(
            roi=Roi((0, 0), (2, 2)),
            dtype=np.float32,
            interpolatable=True,
            voxel_size=(1, 1),
        )
        self.provides(self.a_key, spec)
        self.provides(self.b_key, spec)

    def provide(self, request):
        batch = Batch()

        spec = self.spec[self.a_key]
        spec.roi = request[self.a_key].roi

        batch.arrays[self.a_key] = Array(
            np.array([[0, 1], [2, 3]], dtype=np.float32), spec
        )

        spec = self.spec[self.b_key]
        spec.roi = request[self.b_key].roi

        batch.arrays[self.b_key] = Array(
            np.array([[0, 1], [2, 3]], dtype=np.float32), spec
        )

        return batch


def create_meta_graph(meta_base):
    """

    :param meta_base: Base name (no extension) for meta graph path
    :return:
    """

    def mknet():
        import tensorflow as tf

        # create a tf graph
        a = tf.placeholder(tf.float32, shape=(2, 2))
        b = tf.placeholder(tf.float32, shape=(2, 2))
        v = tf.Variable(1, dtype=tf.float32)
        c = a * b * v

        # dummy "loss"
        loss = tf.norm(c)

        # dummy optimizer
        opt = tf.train.AdamOptimizer()
        optimizer = opt.minimize(loss)

        tf.train.export_meta_graph(filename=meta_base + ".meta")

        with open(meta_base + ".names", "w") as f:
            for x in [a, b, c, optimizer, loss]:
                f.write(x.name + "\n")

    mknet_proc = multiprocessing.Process(target=mknet)
    mknet_proc.start()
    mknet_proc.join()

    with open(meta_base + ".names") as f:
        names = [line.strip("\n") for line in f]

    return names


@pytest.mark.skipif(
    isinstance(tensorflow, NoSuchModule), reason="tensorflow is not installed"
)
def test_output(tmpdir):
    meta_base = tmpdir / "tf_graph"

    a_key = ArrayKey("A")
    b_key = ArrayKey("B")
    c_key = ArrayKey("C")
    a_grad_key = ArrayKey("GRADIENT_A")

    # create model meta graph file and get input/output names
    (a, b, c, optimizer, loss) = create_meta_graph(meta_base)

    source = ExampleTensorflowTrainSource()
    train = Train(
        meta_base,
        optimizer=optimizer,
        loss=loss,
        inputs={a: a_key, b: b_key},
        outputs={c: c_key},
        gradients={a: a_grad_key},
        save_every=100,
    )
    pipeline = source + train

    request = BatchRequest(
        {
            a_key: ArraySpec(roi=Roi((0, 0), (2, 2))),
            b_key: ArraySpec(roi=Roi((0, 0), (2, 2))),
            c_key: ArraySpec(roi=Roi((0, 0), (2, 2))),
            a_grad_key: ArraySpec(roi=Roi((0, 0), (2, 2))),
        }
    )

    # train for a couple of iterations
    with build(pipeline):
        batch = pipeline.request_batch(request)

        assert abs(batch.loss - 9.8994951) < 1e-3

        gradient_a = batch.arrays[a_grad_key].data
        assert gradient_a[0, 0] < gradient_a[0, 1]
        assert gradient_a[0, 1] < gradient_a[1, 0]
        assert gradient_a[1, 0] < gradient_a[1, 1]

        for i in range(200 - 1):
            loss1 = batch.loss
            batch = pipeline.request_batch(request)
            loss2 = batch.loss
            assert loss2 < loss1

    # resume training
    with build(pipeline):
        for i in range(100):
            loss1 = batch.loss
            batch = pipeline.request_batch(request)
            loss2 = batch.loss
            assert loss2 < loss1

    # predict
    # source = ExampleTensorflowTrainSource()
    # predict = Predict(
    # meta_base + '_checkpoint_300',
    # inputs={a: a_key, b: b_key},
    # outputs={c: c_key},
    # max_shared_memory=1024*1024)
    # pipeline = source + predict

    # request = BatchRequest({
    # a_key: ArraySpec(roi=Roi((0, 0), (2, 2))),
    # b_key: ArraySpec(roi=Roi((0, 0), (2, 2))),
    # c_key: ArraySpec(roi=Roi((0, 0), (2, 2))),
    # })

    # with build(pipeline):

    # prev_c = None

    # for i in range(100):
    # batch = pipeline.request_batch(request)
    # c = batch.arrays[c_key].data

    # if prev_c is not None:
    # assert (np.equal(c, prev_c))
    # prev_c = c
