import copy
import random

from .batch_provider import BatchProvider

class RandomProvider(BatchProvider):
    '''Randomly selects one of the upstream providers.
    '''

    def setup(self):

        assert len(self.get_upstream_providers()) > 0, "at least one batch provider needs to be added to the RandomProvider"

        common_spec = None

        # advertise outputs only if all upstream providers have them
        for provider in self.get_upstream_providers():

            if common_spec is None:
                common_spec = copy.deepcopy(provider.spec)
            else:
                for identifier, spec in provider.spec.items():
                    if identifier not in common_spec:
                        del common_spec[identifier]

        for identifier, spec in common_spec.items():
            self.provides(identifier, spec)

    def provide(self, request):
        return random.choice(self.get_upstream_providers()).request_batch(request)
