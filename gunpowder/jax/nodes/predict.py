from gunpowder.array import ArrayKey, Array
from gunpowder.array_spec import ArraySpec
from gunpowder.ext import jax
from gunpowder.nodes.generic_predict import GenericPredict
from gunpowder.jax import GenericJaxModel

import logging
from typing import Dict, Union

logger = logging.getLogger(__name__)


class Predict(GenericPredict):
    """JAX implementation of :class:`gunpowder.nodes.Predict`.

    Args:

        model (subclass of ``gunpowder.jax.GenericJaxModel``):

            The model to use for prediction.

        inputs (``dict``, ``string`` -> :class:`ArrayKey`):

            Dictionary from the names of input tensors in the network to
            array keys.

        outputs (``dict``, ``string`` -> :class:`ArrayKey`):

            Dictionary from the names of output tensors in the network to array
            keys. New arrays will be generated by this node for each entry (if
            requested downstream).

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`, optional):

            Used to set the specs of generated arrays (``outputs``). This is
            useful to set the ``voxel_size``, for example, if they differ from
            the voxel size of the input arrays. Only fields that are not
            ``None`` in the given :class:`ArraySpec` will be used.

        checkpoint: (``string``, optional):

            An optional path to the saved parameters for your jax module.
            These will be loaded and used for prediction if provided.

        spawn_subprocess (bool, optional): Whether to run ``predict`` in a
            separate process. Default is false.
    """

    def __init__(
        self,
        model: GenericJaxModel,
        inputs: Dict[str, ArrayKey],
        outputs: Dict[Union[str, int], ArrayKey],
        array_specs: Dict[ArrayKey, ArraySpec] = None,
        checkpoint: str = None,
        spawn_subprocess=False
    ):

        self.array_specs = array_specs if array_specs is not None else {}

        super(Predict, self).__init__(
            inputs,
            outputs,
            array_specs,
            spawn_subprocess=spawn_subprocess)

        self.model = model
        self.checkpoint = checkpoint
        self.model_params = None

    def start(self):
        if self.checkpoint is not None:
            with open(self.checkpoint, 'rb') as f:
                self.model_params = pickle.load(f)

    def predict(self, batch, request):
        inputs = self.get_inputs(batch)

        if self.model_params is None:
            # need to init model first
            rng = jax.random.PRNGKey(request.random_seed)
            self.model_params = self.model.initialize(rng, inputs)

        out = jax.jit(self.model.forward)(self.model_params, inputs)
        outputs = self.get_outputs(out, request)
        self.update_batch(batch, request, outputs)

    def get_inputs(self, batch):
        model_inputs = {
            key: jax.device_put(batch[value].data)
            for key, value in self.inputs.items()
        }
        return model_inputs

    def get_outputs(self, module_out, request):
        outputs = {}
        for key, value in self.outputs.items():
            if value in request:
                outputs[value] = module_out[key]
        return outputs

    def update_batch(self, batch, request, requested_outputs):
        for array_key, tensor in requested_outputs.items():
            spec = self.spec[array_key].copy()
            spec.roi = request[array_key].roi
            batch.arrays[array_key] = Array(tensor, spec)

    def stop(self):
        pass