import copy
import random

from .batch_provider import BatchProvider

class RandomProvider(BatchProvider):
    '''Randomly selects one of the upstream providers::

        (a + b + c) + RandomProvider()

    will create a provider that randomly relays requests to providers ``a``,
    ``b``, or ``c``. Array and point keys of ``a``, ``b``, and ``c`` should be
    the same.
    '''

    def setup(self):

        assert len(self.get_upstream_providers()) > 0, "at least one batch provider needs to be added to the RandomProvider"

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
        return random.choice(self.get_upstream_providers()).request_batch(request)
