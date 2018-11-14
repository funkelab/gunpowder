import copy
import numpy as np

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

    def __init__(self, probabilities=None):
        self.probabilities = probabilities

        # automatically normalize probabilities to sum to 1
        if self.probabilities is not None:
            self.probabilities = [float(x)/np.sum(probabilities) for x in
                                  self.probabilities]

    def setup(self):

        assert len(self.get_upstream_providers()) > 0, "at least one batch provider needs to be added to the RandomProvider"
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
                for key, spec in provider.spec.items():
                    if key not in common_spec:
                        del common_spec[key]

        for key, spec in common_spec.items():
            self.provides(key, spec)

    def provide(self, request):
        return np.random.choice(self.get_upstream_providers(),
                                p=self.probabilities).request_batch(request)
