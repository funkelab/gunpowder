import random
import copy
from batch_provider import BatchProvider

class RandomProvider(BatchProvider):
    '''Randomly selects one of the upstream providers.
    '''

    def setup(self):
        self.spec = None
        assert len(self.get_upstream_providers()) > 0, "at least one batch provider needs to be added to the RandomProvider"
        for provider in self.get_upstream_providers():
            if self.spec is None:
                self.spec = copy.deepcopy(provider.get_spec())
            else:
                self.spec.has_gt &= provider.get_spec().has_gt
                self.spec.has_gt_mask &= provider.get_spec().has_gt_mask

    def get_spec(self):
        return self.spec

    def request_batch(self, batch_spec):
        return random.choice(self.get_upstream_providers()).request_batch(batch_spec)
