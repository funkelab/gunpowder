from .provider_test import ProviderTest
from gunpowder import (
    BatchProvider,
    Batch,
    BatchRequest,
    ArraySpec,
    ArrayKeys,
    ArrayKey,
    Array,
    Roi,
    Stack,
    build
)
from gunpowder.ext import keras, NoSuchModule
import logging
import numpy as np
from unittest import skipIf


class TestKerasTrainSource(BatchProvider):

    def setup(self):

        spec = ArraySpec(
            roi=Roi((0, 0), (3, 3)),
            dtype=np.float32,
            interpolatable=True,
            voxel_size=(1, 1))
        self.provides(ArrayKeys.X, spec)

        spec = ArraySpec(nonspatial=True)
        self.provides(ArrayKeys.Y, spec)

    def provide(self, request):

        batch = Batch()

        spec = self.spec[ArrayKeys.X]
        spec.roi = request[ArrayKeys.X].roi

        batch.arrays[ArrayKeys.X] = Array(
            np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]], dtype=np.float32),
            spec)

        spec = self.spec[ArrayKeys.Y]

        batch.arrays[ArrayKeys.Y] = Array(
            np.array([0, 1], dtype=np.uint64),
            spec)

        return batch

@skipIf(isinstance(keras, NoSuchModule), "keras is not installed")
class TestKerasTrain(ProviderTest):

    def create_model(self):

        model = keras.Sequential()
        model.add(
            keras.layers.Convolution2D(
                filters=5,
                kernel_size=3,
                activation='relu',
                input_shape=(1, 3, 3),
                data_format='channels_first',
                name='input'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(2, activation='softmax', name='output'))

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

        model.save(self.path_to('model'))

        return ['input', 'output']

    def test_output(self):

        logging.getLogger('gunpowder.keras.nodes.train').setLevel(logging.INFO)

        model_file = self.path_to('tf_graph')

        ArrayKey('X')
        ArrayKey('Y')
        ArrayKey('Y_PREDICTED')

        # create model meta graph file and get input/output names
        (x, y) = self.create_model()

        pipeline = TestKerasTrainSource()
        pipeline += Stack(num_repetitions=10)
        pipeline += keras.Train(
            self.path_to('model'),
            x={x: ArrayKeys.X},
            y={y: ArrayKeys.Y},
            outputs={y: ArrayKeys.Y_PREDICTED},
            array_specs={
                ArrayKeys.Y_PREDICTED: ArraySpec(nonspatial=True)
            },
            save_every=100)

        request = BatchRequest({
            ArrayKeys.X: ArraySpec(roi=Roi((0, 0), (3, 3))),
            ArrayKeys.Y: ArraySpec(nonspatial=True),
            ArrayKeys.Y_PREDICTED: ArraySpec(nonspatial=True)
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
