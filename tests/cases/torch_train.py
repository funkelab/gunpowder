from .helper_sources import ArraySource
from gunpowder import (
    BatchProvider,
    BatchRequest,
    ArraySpec,
    Roi,
    Coordinate,
    ArrayKeys,
    ArrayKey,
    Array,
    Batch,
    Scan,
    PreCache,
    MergeProvider,
    build,
)
from gunpowder.ext import torch, NoSuchModule
from gunpowder.torch import Train, Predict
from unittest import skipIf, expectedFailure
import numpy as np

import logging


# Example 2D source
def example_2d_source(array_key: ArrayKey):
    array_spec = ArraySpec(
        roi=Roi((0, 0), (17, 17)),
        dtype=np.float32,
        interpolatable=True,
        voxel_size=(1, 1),
    )
    data = np.array(list(range(17)), dtype=np.float32).reshape([17, 1])
    data = data + data.T
    array = Array(data, array_spec)
    return ArraySource(array_key, array)


def example_train_source(a_key, b_key, c_key):
    spec1 = ArraySpec(
        roi=Roi((0, 0), (2, 2)),
        dtype=np.float32,
        interpolatable=True,
        voxel_size=(1, 1),
    )
    spec2 = ArraySpec(nonspatial=True)

    data1 = np.array([[0, 1], [2, 3]], dtype=np.float32)
    data2 = np.array([1], dtype=np.float32)

    source_a = ArraySource(a_key, Array(data1, spec1))
    source_b = ArraySource(b_key, Array(data1, spec1))
    source_c = ArraySource(c_key, Array(data2, spec2))

    return (source_a, source_b, source_c) + MergeProvider()


if torch is not NoSuchModule:

    class ExampleLinearModel(torch.nn.Module):
        def __init__(self):
            super(ExampleLinearModel, self).__init__()
            self.linear = torch.nn.Linear(4, 1, False)
            self.linear.weight.data = torch.Tensor([0, 1, 2, 3])

        def forward(self, a, b):
            a = a.reshape(-1)
            b = b.reshape(-1)
            c_pred = self.linear(a * b)
            d_pred = c_pred * 2
            return d_pred


@skipIf(isinstance(torch, NoSuchModule), "torch is not installed")
def test_loss_drops(tmpdir):
    checkpoint_basename = str(tmpdir / "model")

    a_key = ArrayKey("A")
    b_key = ArrayKey("B")
    c_key = ArrayKey("C")
    c_predicted_key = ArrayKey("C_PREDICTED")
    c_gradient_key = ArrayKey("C_GRADIENT")

    model = ExampleLinearModel()
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-7, momentum=0.999)

    source = example_train_source(a_key, b_key, c_key)
    train = Train(
        model=model,
        optimizer=optimizer,
        loss=loss,
        inputs={"a": a_key, "b": b_key},
        loss_inputs={0: c_predicted_key, 1: c_key},
        outputs={0: c_predicted_key},
        gradients={0: c_gradient_key},
        array_specs={
            c_predicted_key: ArraySpec(nonspatial=True),
            c_gradient_key: ArraySpec(nonspatial=True),
        },
        checkpoint_basename=checkpoint_basename,
        save_every=100,
        spawn_subprocess=False,
    )
    pipeline = source + train

    request = BatchRequest(
        {
            a_key: ArraySpec(roi=Roi((0, 0), (2, 2))),
            b_key: ArraySpec(roi=Roi((0, 0), (2, 2))),
            c_key: ArraySpec(nonspatial=True),
            c_predicted_key: ArraySpec(nonspatial=True),
            c_gradient_key: ArraySpec(nonspatial=True),
        }
    )

    # train for a couple of iterations
    with build(pipeline):
        batch = pipeline.request_batch(request)

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


@skipIf(isinstance(torch, NoSuchModule), "torch is not installed")
def test_output():
    logging.getLogger("gunpowder.torch.nodes.predict").setLevel(logging.INFO)

    a_key = ArrayKey("A")
    b_key = ArrayKey("B")
    c_key = ArrayKey("C")
    c_pred = ArrayKey("C_PREDICTED")
    d_pred = ArrayKey("D_PREDICTED")

    model = ExampleLinearModel()

    source = example_train_source(a_key, b_key, c_key)
    predict = Predict(
        model=model,
        inputs={"a": a_key, "b": b_key},
        outputs={"linear": c_pred, 0: d_pred},
        array_specs={
            c_key: ArraySpec(nonspatial=True),
            c_pred: ArraySpec(nonspatial=True),
            d_pred: ArraySpec(nonspatial=True),
        },
        spawn_subprocess=True,
    )
    pipeline = source + predict

    request = BatchRequest(
        {
            a_key: ArraySpec(roi=Roi((0, 0), (2, 2))),
            b_key: ArraySpec(roi=Roi((0, 0), (2, 2))),
            c_key: ArraySpec(nonspatial=True),
            c_pred: ArraySpec(nonspatial=True),
            d_pred: ArraySpec(nonspatial=True),
        }
    )

    # train for a couple of iterations
    with build(pipeline):
        batch1 = pipeline.request_batch(request)
        batch2 = pipeline.request_batch(request)

        assert np.isclose(batch1[c_pred].data, batch2[c_pred].data)
        assert np.isclose(batch1[c_pred].data, 1 + 4 * 2 + 9 * 3)
        assert np.isclose(batch2[d_pred].data, 2 * (1 + 4 * 2 + 9 * 3))


if torch is not NoSuchModule:

    class Example2DModel(torch.nn.Module):
        def __init__(self):
            super(Example2DModel, self).__init__()
            self.linear = torch.nn.Conv2d(1, 1, 3)

        def forward(self, a):
            a = a.unsqueeze(0).unsqueeze(0)
            pred = self.linear(a)
            a = a.squeeze(0).squeeze(0)
            pred = pred.squeeze(0).squeeze(0)
            return pred


@skipIf(isinstance(torch, NoSuchModule), "torch is not installed")
def test_scan():
    logging.getLogger("gunpowder.torch.nodes.predict").setLevel(logging.INFO)

    a_key = ArrayKey("A")
    pred = ArrayKey("PRED")

    model = Example2DModel()

    reference_request = BatchRequest()
    reference_request[a_key] = ArraySpec(roi=Roi((0, 0), (7, 7)))
    reference_request[pred] = ArraySpec(roi=Roi((1, 1), (5, 5)))

    source = example_2d_source(a_key)
    predict = Predict(
        model=model,
        inputs={"a": a_key},
        outputs={0: pred},
        array_specs={pred: ArraySpec()},
    )
    pipeline = source + predict + Scan(reference_request, num_workers=2)

    request = BatchRequest(
        {
            a_key: ArraySpec(roi=Roi((0, 0), (17, 17))),
            pred: ArraySpec(roi=Roi((0, 0), (15, 15))),
        }
    )

    # train for a couple of iterations
    with build(pipeline):
        batch = pipeline.request_batch(request)
        assert pred in batch


def test_precache():
    logging.getLogger("gunpowder.torch.nodes.predict").setLevel(logging.INFO)

    a_key = ArrayKey("A")
    pred = ArrayKey("PRED")

    model = Example2DModel()

    reference_request = BatchRequest()
    reference_request[a_key] = ArraySpec(roi=Roi((0, 0), (7, 7)))
    reference_request[pred] = ArraySpec(roi=Roi((1, 1), (5, 5)))

    source = example_2d_source(a_key)
    predict = Predict(
        model=model,
        inputs={"a": a_key},
        outputs={0: pred},
        array_specs={pred: ArraySpec()},
    )
    pipeline = source + predict + PreCache(cache_size=3, num_workers=2)

    request = BatchRequest(
        {
            a_key: ArraySpec(roi=Roi((0, 0), (17, 17))),
            pred: ArraySpec(roi=Roi((0, 0), (15, 15))),
        }
    )

    # train for a couple of iterations
    with build(pipeline):
        batch = pipeline.request_batch(request)
        assert pred in batch
