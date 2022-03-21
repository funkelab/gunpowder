import copy
import numpy as np

from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec

from .batch_provider import BatchProvider


class RandomProvider(BatchProvider):
    '''Randomly selects one of the upstream providers::

        (a + b + c) + RandomProvider()

    will create a provider that randomly relays requests to providers ``a``,
    ``b``, or ``c``. Array and point keys of ``a``, ``b``, and ``c`` should be
    the same.

    Args:
        probabilities (1-D array-like, optional): An optional list of
            probabilities for choosing upstream providers, given in the
            same order. Probabilities do not need to be normalized. Default
            is ``None``, corresponding to equal probabilities.
    '''

    def __init__(self, probabilities=None, randomness_store_key=None):
        self.probabilities = probabilities
        self.randomness_store_key = randomness_store_key

        # automatically normalize probabilities to sum to 1
        if self.probabilities is not None:
            self.probabilities = [float(x)/np.sum(probabilities) for x in
                                  self.probabilities]

    def setup(self):
        self.enable_placeholders()
        assert len(self.get_upstream_providers()) > 0,\
            "at least one batch provider must be added to the RandomProvider"
        if self.probabilities is not None:
            assert len(self.get_upstream_providers()) == len(
                self.probabilities), "if probabilities are specified, they " \
                                     "need to be given for each batch " \
                                     "provider added to the RandomProvider"

        common_spec = None

        # advertise outputs only if all upstream providers have them
        for provider in self.get_upstream_providers():

            if common_spec is None:
                common_spec = copy.deepcopy(provider.spec)
            else:
                for key, spec in list(common_spec.items()):
                    if key not in provider.spec:
                        del common_spec[key]

        for key, spec in common_spec.items():
            self.provides(key, spec)

        if self.randomness_store_key is not None:
            self.provides(self.randomness_store_key, ArraySpec(nonspatial=True))

    def provide(self, request):
        # Random seed is set in provide rather than prepare since this node
        # is not a batch filter
        np.random.seed(request.random_seed)

        if self.randomness_store_key is not None:
            del request[self.randomness_store_key]

        i = np.random.choice(
            range(len(self.get_upstream_providers())), p=self.probabilities
        )
        provider = self.get_upstream_providers()[i]
        batch = provider.request_batch(request)
        if self.randomness_store_key is not None:
            batch[self.randomness_store_key] = Array(
                np.array(i), ArraySpec(nonspatial=True)
            )
        return batch
