class GenericJaxModel:
    """An interface for models to follow in order to train or predict. A model
    implementing this interface will need to contain not only the forward
    model but also loss and update fn. Some examples can be found in
    https://github.com/funkelab/funlib.learn.jax

    Args:

        is_training (``bool``):

            Indicating whether the model will be used for training
            or inferencing.
    """

    def __init__(self, is_training):
        pass

    def initialize(self, rng_key, inputs):
        """Initialize parameters for training.

        Args:

            rng_key (jax.random.PRNGKey):

                Seed for parameter initialization

            inputs (``dict``, ``string`` -> jnp.ndarray):

                Dictionary of inputs, provided to initialize parameters
                with the correct dimensions.

        Return:

            params (Any):

                Function should return an object encapsulating different
                parameters of the model.
        """
        raise RuntimeError("Unimplemented")

    def forward(self, params, inputs):
        """Run the forward model.

        Args:

            params (Any):

                Model parameters.

            inputs (``dict``, ``string`` -> jnp.ndarray):

                Dictionary of inputs.

        Return:

            outputs (``dict``, ``string`` -> jnp.ndarray):

                Dictionary of outputs.
        """
        raise RuntimeError("Unimplemented")

    def train_step(self, params, inputs, pmapped):
        """Run one iteration of training on the model.

        Args:

            params (Any):

                Model parameters.

            inputs (``dict``, ``string`` -> jnp.ndarray):

                Dictionary of inputs.

            pmapped (``bool``):

                Whether the function is run with `jax.pmap` or not.
                If pmapped across devices, the function should take care to
                synchronize gradients during the train step.
                The `axis_name` is set to the ``string`` "num_devices"

        Return:

            Tuple(new_params, outputs, loss)

                new_params (Any):

                    Updated model parameters.

                outputs (``dict``, ``string`` -> jnp.ndarray):

                    Dictionary of outputs.

                loss (Union[``float``, (``dict``, ``string`` -> ``float``)]):

                    Loss value of this iteration. Value can either be a single
                    ``float`` or a dictionary of multiple losses.
        """
        raise RuntimeError("Unimplemented")
