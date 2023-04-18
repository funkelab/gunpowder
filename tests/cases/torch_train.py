from .provider_test import ProviderTest
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
    build,
)
from gunpowder.ext import torch, NoSuchModule
from gunpowder.torch import Train, Predict
from unittest import skipIf, expectedFailure
import numpy as np

import logging


class ExampleTorchTrain2DSource(BatchProvider):
    def __init__(self):
        pass

    def setup(self):
        spec = ArraySpec(
            roi=Roi((0, 0), (17, 17)),
            dtype=np.float32,
            interpolatable=True,
            voxel_size=(1, 1),
        )
        self.provides(ArrayKeys.A, spec)

    def provide(self, request):
        batch = Batch()

        spec = self.spec[ArrayKeys.A]

        x = np.array(list(range(17)), dtype=np.float32).reshape([17, 1])
        x = x + x.T

        batch.arrays[ArrayKeys.A] = Array(x, spec).crop(request[ArrayKeys.A].roi)

        return batch


class ExampleTorchTrainSource(BatchProvider):
    def setup(self):
        spec = ArraySpec(
            roi=Roi((0, 0), (2, 2)),
            dtype=np.float32,
            interpolatable=True,
            voxel_size=(1, 1),
        )
        self.provides(ArrayKeys.A, spec)
        self.provides(ArrayKeys.B, spec)

        spec = ArraySpec(nonspatial=True)
        self.provides(ArrayKeys.C, spec)

    def provide(self, request):
        batch = Batch()

        spec = self.spec[ArrayKeys.A]
        spec.roi = request[ArrayKeys.A].roi

        batch.arrays[ArrayKeys.A] = Array(
            np.array([[0, 1], [2, 3]], dtype=np.float32), spec
        )

        spec = self.spec[ArrayKeys.B]
        spec.roi = request[ArrayKeys.B].roi

        batch.arrays[ArrayKeys.B] = Array(
            np.array([[0, 1], [2, 3]], dtype=np.float32), spec
        )

        spec = self.spec[ArrayKeys.C]

        batch.arrays[ArrayKeys.C] = Array(np.array([1], dtype=np.float32), spec)

        return batch


@skipIf(isinstance(torch, NoSuchModule), "torch is not installed")
class TestTorchTrain(ProviderTest):
    def test_output(self):
        logging.getLogger("gunpowder.torch.nodes.train").setLevel(logging.INFO)

        checkpoint_basename = self.path_to("model")

        ArrayKey("A")
        ArrayKey("B")
        ArrayKey("C")
        ArrayKey("C_PREDICTED")
        ArrayKey("C_GRADIENT")

        class ExampleModel(torch.nn.Module):
            def __init__(self):
                super(ExampleModel, self).__init__()
                self.linear = torch.nn.Linear(4, 1, False)

            def forward(self, a, b):
                a = a.reshape(-1)
                b = b.reshape(-1)
                return self.linear(a * b)

        model = ExampleModel()
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-7, momentum=0.999)

        source = ExampleTorchTrainSource()
        train = Train(
            model=model,
            optimizer=optimizer,
            loss=loss,
            inputs={"a": ArrayKeys.A, "b": ArrayKeys.B},
            loss_inputs={0: ArrayKeys.C_PREDICTED, 1: ArrayKeys.C},
            outputs={0: ArrayKeys.C_PREDICTED},
            gradients={0: ArrayKeys.C_GRADIENT},
            array_specs={
                ArrayKeys.C_PREDICTED: ArraySpec(nonspatial=True),
                ArrayKeys.C_GRADIENT: ArraySpec(nonspatial=True),
            },
            checkpoint_basename=checkpoint_basename,
            save_every=100,
            spawn_subprocess=True,
        )
        pipeline = source + train

        request = BatchRequest(
            {
                ArrayKeys.A: ArraySpec(roi=Roi((0, 0), (2, 2))),
                ArrayKeys.B: ArraySpec(roi=Roi((0, 0), (2, 2))),
                ArrayKeys.C: ArraySpec(nonspatial=True),
                ArrayKeys.C_PREDICTED: ArraySpec(nonspatial=True),
                ArrayKeys.C_GRADIENT: ArraySpec(nonspatial=True),
            }
        )

        # train for a couple of iterations
        with build(pipeline):
            batch = pipeline.request_batch(request)

            for i in range(200 - 1):
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


@skipIf(isinstance(torch, NoSuchModule), "torch is not installed")
class TestTorchPredict(ProviderTest):
    def test_output(self):
        logging.getLogger("gunpowder.torch.nodes.predict").setLevel(logging.INFO)

        a = ArrayKey("A")
        b = ArrayKey("B")
        c = ArrayKey("C")
        c_pred = ArrayKey("C_PREDICTED")
        d_pred = ArrayKey("D_PREDICTED")

        class ExampleModel(torch.nn.Module):
            def __init__(self):
                super(ExampleModel, self).__init__()
                self.linear = torch.nn.Linear(4, 1, False)
                self.linear.weight.data = torch.Tensor([1, 1, 1, 1])

            def forward(self, a, b):
                a = a.reshape(-1)
                b = b.reshape(-1)
                c_pred = self.linear(a * b)
                d_pred = c_pred * 2
                return d_pred

        model = ExampleModel()

        source = ExampleTorchTrainSource()
        predict = Predict(
            model=model,
            inputs={"a": a, "b": b},
            outputs={"linear": c_pred, 0: d_pred},
            array_specs={
                c: ArraySpec(nonspatial=True),
                c_pred: ArraySpec(nonspatial=True),
                d_pred: ArraySpec(nonspatial=True),
            },
            spawn_subprocess=True,
        )
        pipeline = source + predict

        request = BatchRequest(
            {
                a: ArraySpec(roi=Roi((0, 0), (2, 2))),
                b: ArraySpec(roi=Roi((0, 0), (2, 2))),
                c: ArraySpec(nonspatial=True),
                c_pred: ArraySpec(nonspatial=True),
                d_pred: ArraySpec(nonspatial=True),
            }
        )

        # train for a couple of iterations
        with build(pipeline):
            batch1 = pipeline.request_batch(request)
            batch2 = pipeline.request_batch(request)

            assert np.isclose(batch1[c_pred].data, batch2[c_pred].data)
            assert np.isclose(batch1[c_pred].data, 1 + 4 + 9)
            assert np.isclose(batch2[d_pred].data, 2 * (1 + 4 + 9))


if not isinstance(torch, NoSuchModule):

    class ExampleModel(torch.nn.Module):
        def __init__(self):
            super(ExampleModel, self).__init__()
            self.linear = torch.nn.Conv2d(1, 1, 3)

        def forward(self, a):
            a = a.unsqueeze(0).unsqueeze(0)
            pred = self.linear(a)
            a = a.squeeze(0).squeeze(0)
            pred = pred.squeeze(0).squeeze(0)
            return pred


@skipIf(isinstance(torch, NoSuchModule), "torch is not installed")
class TestTorchPredictMultiprocessing(ProviderTest):
    def test_scan(self):
        logging.getLogger("gunpowder.torch.nodes.predict").setLevel(logging.INFO)

        a = ArrayKey("A")
        pred = ArrayKey("PRED")

        model = ExampleModel()

        reference_request = BatchRequest()
        reference_request[a] = ArraySpec(roi=Roi((0, 0), (7, 7)))
        reference_request[pred] = ArraySpec(roi=Roi((1, 1), (5, 5)))

        source = ExampleTorchTrain2DSource()
        predict = Predict(
            model=model,
            inputs={"a": a},
            outputs={0: pred},
            array_specs={pred: ArraySpec()},
        )
        pipeline = source + predict + Scan(reference_request, num_workers=2)

        request = BatchRequest(
            {
                a: ArraySpec(roi=Roi((0, 0), (17, 17))),
                pred: ArraySpec(roi=Roi((0, 0), (15, 15))),
            }
        )

        # train for a couple of iterations
        with build(pipeline):
            batch = pipeline.request_batch(request)
            assert pred in batch

    def test_precache(self):
        logging.getLogger("gunpowder.torch.nodes.predict").setLevel(logging.INFO)

        a = ArrayKey("A")
        pred = ArrayKey("PRED")

        model = ExampleModel()

        reference_request = BatchRequest()
        reference_request[a] = ArraySpec(roi=Roi((0, 0), (7, 7)))
        reference_request[pred] = ArraySpec(roi=Roi((1, 1), (5, 5)))

        source = ExampleTorchTrain2DSource()
        predict = Predict(
            model=model,
            inputs={"a": a},
            outputs={0: pred},
            array_specs={pred: ArraySpec()},
        )
        pipeline = source + predict + PreCache(cache_size=3, num_workers=2)

        request = BatchRequest(
            {
                a: ArraySpec(roi=Roi((0, 0), (17, 17))),
                pred: ArraySpec(roi=Roi((0, 0), (15, 15))),
            }
        )

        # train for a couple of iterations
        with build(pipeline):
            batch = pipeline.request_batch(request)
            assert pred in batch
