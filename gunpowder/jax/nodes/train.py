import logging
import numpy as np
import jax
import jax.numpy as jnp
import pickle
import os

from gunpowder.array import ArrayKey, Array
from gunpowder.array_spec import ArraySpec
from gunpowder.ext import tensorboardX, NoSuchModule
from gunpowder.nodes.generic_train import GenericTrain
from gunpowder.jax import GenericJaxModel

from typing import Dict, Union, Optional


logger = logging.getLogger(__name__)


class Train(GenericTrain):
    """JAX implementation of :class:`gunpowder.nodes.GenericTrain`.

    Args:

        model (subclass of ``gunpowder.jax.GenericJaxModel``):

            The model to train. This model encapsulates the forward model,
            loss, and optimizer.

        inputs (``dict``, ``string`` -> Union[np.ndarray, ArrayKey]):

            Dictionary from the names of input tensors expected by the
            ``train_step`` method to array keys or ndarray.

        outputs (``dict``, ``string`` -> :class:`ArrayKey`):

            Dictionary from the names of tensors in the network to array
            keys. If the key is a string, the tensor will be retrieved
            by checking the model for an attribute with they key as its name.
            If the key is an integer, it is interpreted as a tuple index of
            the outputs of the network.
            New arrays will be generated by this node for each entry (if
            requested downstream).

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`, optional):

            Used to set the specs of generated arrays (at the moment only
            ``output``). This is useful to set the ``voxel_size``, for example,
            if they differ from the voxel size of the input arrays. Only fields
            that are not ``None`` in the given :class:`ArraySpec` will be used.

        checkpoint_basename (``string``, optional):

            The basename used for checkpoint files. Defaults to ``model``.

        save_every (``int``, optional):

            After how many iterations to create a checkpoint to store the
            learnt weights.

        keep_n_checkpoints (``int``, optional):

            Number of checkpoints to keep. Node will attempt to delete older
            checkpoints. Default is `None` (no deletion).

        log_dir (``string``, optional):

            Directory for saving tensorboard summaries.

        log_every (``int``, optional):

            After how many iterations to write out tensorboard summaries.

        spawn_subprocess (``bool``, optional):

            Whether to run the ``train_step`` in a separate process. Default is
            false.

        n_devices (``int``, optional):

            Number of GPU devices to train on concurrently using `jax.pmap`. If
            `None`, the number of available GPUs will be automatically detected
            and used.
    """

    def __init__(
        self,
        model: GenericJaxModel,
        inputs: Dict[str, Union[np.ndarray, ArrayKey]],
        outputs: Dict[Union[int, str], ArrayKey],
        gradients: Dict[Union[int, str], ArrayKey] = {},
        array_specs: Optional[Dict[ArrayKey, ArraySpec]] = None,
        checkpoint_basename: str = "model",
        save_every: int = 2000,
        keep_n_checkpoints: Optional[int] = None,
        log_dir: str = None,
        log_every: int = 1,
        spawn_subprocess: bool = False,
        n_devices: Optional[int] = None
    ):

        # not yet implemented
        gradients = gradients

        super(Train, self).__init__(
            inputs, outputs, gradients, array_specs,
            spawn_subprocess=spawn_subprocess
        )

        self.model = model
        self.checkpoint_basename = checkpoint_basename
        self.save_every = save_every
        if n_devices is None:
            n_devices = jax.local_device_count()  # autodetect available GPUs
        self.n_devices = n_devices
        self.local_devices = jax.local_devices()
        self.keep_n_checkpoints = keep_n_checkpoints

        self.iteration = 0

        if not isinstance(tensorboardX, NoSuchModule) and log_dir is not None:
            self.summary_writer = tensorboardX.SummaryWriter(log_dir)
            self.log_every = log_every
        else:
            self.summary_writer = None
            if log_dir is not None:
                logger.warning(
                    "log_dir given, but tensorboardX is not installed")

        self.intermediate_layers = {}

    def replicate_params(self, params):
        return jax.tree_map(lambda x: jnp.array([x] * self.n_devices), params)

    def start(self):
        checkpoint, self.iteration = self._get_latest_checkpoint(
            self.checkpoint_basename)

        if checkpoint is not None:

            logger.info("Resuming training from iteration %d", self.iteration)

            with open(checkpoint, 'rb') as f:
                self.model_params = pickle.load(f)
                if self.n_devices > 1:
                    self.model_params = self.replicate_params(self.model_params)
        else:

            logger.info("Starting training from scratch")
            self.model_params = None

    def split_inputs(self, inputs):
        for k, arr in inputs.items():
            assert arr.shape[0] % self.n_devices == 0, (
                f"Batch size should be evenly divisible by the number of "
                f"devices. Input array shape is {arr.shape} but n_device is"
                f" {self.n_devices}")
            inputs[k] = arr.reshape(
                self.n_devices, arr.shape[0] // self.n_devices, *arr.shape[1:])
            inputs[k] = [x for x in inputs[k]]  # make a sequence for put_sharded
        return inputs

    def unstack_device_outputs(self, outputs):
        for k, arr in outputs.items():
            outputs[k] = arr.reshape(arr.shape[0]*arr.shape[1], *arr.shape[2:])
        return outputs

    def train_step(self, batch, request):
        inputs = self.__collect_provided_inputs(batch)
        if self.n_devices > 1:
            inputs = self.split_inputs(inputs)

        # put to device for max performance
        if self.n_devices > 1:
            for k, v in inputs.items():
                inputs[k] = jax.device_put_sharded(v, jax.local_devices())
        else:
            for k, v in inputs.items():
                inputs[k] = jax.device_put(v)

        # initialize model if necessary
        if self.model_params is None:
            # Using a random key is meant to make training reproducible but
            # since gunpowder is not deterministic we will leave it hard-
            # coded for now until gunpowder has the ability to set random seed
            rng = jax.random.PRNGKey(42)
            if self.n_devices > 1:
                rng = jnp.broadcast_to(rng, (self.n_devices,) + rng.shape)
                self.model_params = jax.pmap(self.model.initialize)(rng, inputs)
            else:
                self.model_params = self.model.initialize(rng, inputs)

        requested_outputs = self.__collect_requested_outputs(request)

        if self.n_devices > 1:
            self.model_params, outputs, loss = jax.pmap(
                                            self.model.train_step,
                                            axis_name='num_devices',
                                            donate_argnums=(0,),
                                            static_broadcasted_argnums=(2,))(
                                    self.model_params, inputs, True)
            loss = loss.mean()
            outputs = self.unstack_device_outputs(outputs)  # stack by batch
        else:
            self.model_params, outputs, loss = jax.jit(
                                            self.model.train_step,
                                            donate_argnums=(0,),
                                            static_argnums=(2,))(
                                    self.model_params, inputs, False)

        logger.debug(
            "model outputs: %s",
            {k: v.shape for k, v in outputs.items()})

        # add requested model outputs to batch
        for array_key, array_name in requested_outputs.items():
            spec = self.spec[array_key].copy()
            spec.roi = request[array_key].roi
            batch.arrays[array_key] = Array(
                outputs[array_name], spec
            )

        batch.loss = loss
        self.iteration += 1
        batch.iteration = self.iteration

        if batch.iteration % self.save_every == 0:

            checkpoint_name = self._checkpoint_name(
                self.checkpoint_basename, batch.iteration)

            logger.info("Creating checkpoint %s", checkpoint_name)

            model_state = self.model_params
            if self.n_devices > 1:
                # get only a single copy of param for saving
                model_state = jax.tree_map(lambda x: x[0], model_state)
            with open(checkpoint_name, 'wb') as f:
                pickle.dump(model_state, f)

            if self.keep_n_checkpoints:
                checkpoint_name = self._checkpoint_name(
                    self.checkpoint_basename,
                    batch.iteration - self.keep_n_checkpoints*self.save_every)
                try:
                    os.remove(checkpoint_name)
                    logger.info("Removed checkpoint %s", checkpoint_name)
                except FileNotFoundError:
                    pass

        if self.summary_writer and batch.iteration % self.log_every == 0:
            self.summary_writer.add_scalar("loss", batch.loss, batch.iteration)

    def __collect_requested_outputs(self, request):

        array_outputs = {}

        for output_name, array_key in self.outputs.items():
            if array_key in request:
                array_outputs[array_key] = output_name

        return array_outputs

    def __collect_provided_inputs(self, batch):

        return self.__collect_provided_arrays(self.inputs, batch)

    def __collect_provided_arrays(self, reference, batch):

        arrays = {}

        for array_name, array_key in reference.items():
            if isinstance(array_key, ArrayKey):
                msg = f"batch does not contain {array_key}, array {array_name} will not be set"
                if array_key in batch.arrays:
                    arrays[array_name] = batch.arrays[array_key].data
                else:
                    logger.warn(msg)
            elif isinstance(array_key, np.ndarray):
                arrays[array_name] = array_key
            else:
                raise Exception(
                    "Unknown network array key {}, can't be given to "
                    "network".format(array_key)
                )

        return arrays
