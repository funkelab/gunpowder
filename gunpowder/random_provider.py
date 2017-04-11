import random
from batch_provider import BatchProvider

class RandomProvider(BatchProvider):
    '''Randomly selects one of the upstream providers.
    '''

    def request_batch(self, batch_spec):
        return random.choice(self.get_upstream_providers()).request_batch(batch_spec)
