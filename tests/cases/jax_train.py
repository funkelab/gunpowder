from .provider_test import ProviderTest
from gunpowder import (
    BatchProvider,
    BatchRequest,
    ArraySpec,
    Roi,
    ArrayKeys,
    ArrayKey,
    Array,
    Batch,
    build,
)
from gunpowder.ext import jax, haiku, optax, NoSuchModule
from gunpowder.jax import Train, Predict, GenericJaxModel
from unittest import skipIf
import numpy as np

import logging

# use CPU for JAX tests and avoid GPU compatibility
if not isinstance(jax, NoSuchModule):
    jax.config.update("jax_platform_name", "cpu")


class ExampleJaxTrain2DSource(BatchProvider):
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


class ExampleJaxTrainSource(BatchProvider):
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


@skipIf(isinstance(jax, NoSuchModule), "Jax is not installed")
class TestJaxTrain(ProviderTest):
    def test_output(self):
        logging.getLogger("gunpowder.jax.nodes.train").setLevel(logging.INFO)

        checkpoint_basename = self.path_to("model")

        ArrayKey("A")
        ArrayKey("B")
        ArrayKey("C")
        ArrayKey("C_PREDICTED")
        ArrayKey("C_GRADIENT")

        class ExampleModel(GenericJaxModel):
            def __init__(self, is_training):
                super().__init__(is_training)

                def _linear(x):
                    return haiku.Linear(1, False)(x)

                self.linear = haiku.without_apply_rng(haiku.transform(_linear))
                self.opt = optax.sgd(learning_rate=1e-7, momentum=0.999)

            def initialize(self, rng_key, inputs):
                a = inputs["a"].reshape(-1)
                b = inputs["b"].reshape(-1)
                weight = self.linear.init(rng_key, a * b)
                opt_state = self.opt.init(weight)
                return (weight, opt_state)

            def forward(self, params, inputs):
                a = inputs["a"].reshape(-1)
                b = inputs["b"].reshape(-1)
                return {"c": self.linear.apply(params[0], a * b)}

            def _loss_fn(self, weight, a, b, c):
                c_pred = self.linear.apply(weight, a * b)
                loss = optax.l2_loss(predictions=c_pred, targets=c) * 2
                loss_mean = loss.mean()
                return loss_mean, (c_pred, loss, loss_mean)

            def _apply_optimizer(self, params, grads):
                updates, new_opt_state = self.opt.update(grads, params[1])
                new_weight = optax.apply_updates(params[0], updates)
                return new_weight, new_opt_state

            def train_step(self, params, inputs, pmapped=False):
                a = inputs["a"].reshape(-1)
                b = inputs["b"].reshape(-1)
                c = inputs["c"].reshape(-1)

                grads, (c_pred, loss, loss_mean) = jax.grad(
                    self._loss_fn, has_aux=True
                )(params[0], a, b, c)

                new_weight, new_opt_state = self._apply_optimizer(params, grads)
                new_params = (new_weight, new_opt_state)

                outputs = {
                    "c_pred": c_pred,
                    "grad": loss,
                }
                return new_params, outputs, loss_mean

        model = ExampleModel(is_training=False)

        source = ExampleJaxTrainSource()
        train = Train(
            model=model,
            inputs={"a": ArrayKeys.A, "b": ArrayKeys.B, "c": ArrayKeys.C},
            outputs={"c_pred": ArrayKeys.C_PREDICTED, "grad": ArrayKeys.C_GRADIENT},
            array_specs={
                ArrayKeys.C_PREDICTED: ArraySpec(nonspatial=True),
                ArrayKeys.C_GRADIENT: ArraySpec(nonspatial=True),
            },
            checkpoint_basename=checkpoint_basename,
            save_every=100,
            spawn_subprocess=True,
            n_devices=1,
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


@skipIf(isinstance(jax, NoSuchModule), "Jax is not installed")
class TestJaxPredict(ProviderTest):
    def test_output(self):
        logging.getLogger("gunpowder.jax.nodes.predict").setLevel(logging.INFO)

        a = ArrayKey("A")
        b = ArrayKey("B")
        c = ArrayKey("C")
        c_pred = ArrayKey("C_PREDICTED")
        d_pred = ArrayKey("D_PREDICTED")

        class ExampleModel(GenericJaxModel):
            def __init__(self, is_training):
                super().__init__(is_training)

                def _linear(x):
                    return haiku.Linear(1, False)(x)

                self.linear = haiku.without_apply_rng(haiku.transform(_linear))

            def initialize(self, rng_key, inputs):
                a = inputs["a"].reshape(-1)
                b = inputs["b"].reshape(-1)
                weight = self.linear.init(rng_key, a * b)
                weight["linear"]["w"] = (
                    weight["linear"]["w"].at[:].set(np.array([[1], [1], [1], [1]]))
                )
                return weight

            def forward(self, params, inputs):
                a = inputs["a"].reshape(-1)
                b = inputs["b"].reshape(-1)
                c_pred = self.linear.apply(params, a * b)
                d_pred = c_pred * 2
                return {"c": c_pred, "d": d_pred}

        model = ExampleModel(is_training=False)

        source = ExampleJaxTrainSource()
        predict = Predict(
            model=model,
            inputs={"a": a, "b": b},
            outputs={"c": c_pred, "d": d_pred},
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
