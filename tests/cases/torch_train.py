from .provider_test import ProviderTest
from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import Train
import numpy as np

class TestTorchTrainSource(BatchProvider):

    def setup(self):

        spec = ArraySpec(
            roi=Roi((0, 0), (2, 2)),
            dtype=np.float32,
            interpolatable=True,
            voxel_size=(1, 1))
        self.provides(ArrayKeys.A, spec)
        self.provides(ArrayKeys.B, spec)

        spec = ArraySpec(nonspatial=True)
        self.provides(ArrayKeys.C, spec)

    def provide(self, request):

        batch = Batch()

        spec = self.spec[ArrayKeys.A]
        spec.roi = request[ArrayKeys.A].roi

        batch.arrays[ArrayKeys.A] = Array(
            np.array([[0, 1], [2, 3]], dtype=np.float32),
            spec)

        spec = self.spec[ArrayKeys.B]
        spec.roi = request[ArrayKeys.B].roi

        batch.arrays[ArrayKeys.B] = Array(
            np.array([[0, 1], [2, 3]], dtype=np.float32),
            spec)

        spec = self.spec[ArrayKeys.C]

        batch.arrays[ArrayKeys.C] = Array(
            np.array([1], dtype=np.float32),
            spec)

        return batch

class TestModel(torch.nn.Module):

    def __init__(self):
        super(TestModel, self).__init__()
        self.linear = torch.nn.Linear(4, 1, False)

    def forward(self, a, b):
        a = a.reshape(-1)
        b = b.reshape(-1)
        return self.linear(a*b)

class TestTorchTrain(ProviderTest):

    def test_output(self):

        logging.getLogger('gunpowder.torch.nodes.train').setLevel(logging.INFO)

        checkpoint_basename = self.path_to('model')

        ArrayKey('A')
        ArrayKey('B')
        ArrayKey('C')
        ArrayKey('C_PREDICTED')

        model = TestModel()
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=1e-7,
            momentum=0.999)

        source = TestTorchTrainSource()
        train = Train(
            model=model,
            optimizer=optimizer,
            loss=loss,
            inputs={'a': ArrayKeys.A, 'b': ArrayKeys.B},
            target=ArrayKeys.C,
            output=ArrayKeys.C_PREDICTED,
            checkpoint_basename=checkpoint_basename,
            save_every=100)
        pipeline = source + train

        request = BatchRequest({
            ArrayKeys.A: ArraySpec(roi=Roi((0, 0), (2, 2))),
            ArrayKeys.B: ArraySpec(roi=Roi((0, 0), (2, 2))),
            ArrayKeys.C: ArraySpec(nonspatial=True),
            ArrayKeys.C_PREDICTED: ArraySpec(nonspatial=True),
        })

        # train for a couple of iterations
        with build(pipeline):

            batch = pipeline.request_batch(request)

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
