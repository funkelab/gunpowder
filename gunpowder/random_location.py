from batch_filter import BatchFilter
from random import randint

class RandomLocation(BatchFilter):
    '''Choses a batch at a random location in the bounding box of the upstream 
    provider.
    '''

    def initialize(self):

        provider_spec = self.get_upstream_provider().get_spec()
        if provider_spec.get_bounding_box() is None:
            raise RuntimeError("Can not draw random samples from a provider that does not have a bounding box.")

        self.bounding_box = provider_spec.get_bounding_box()

    def prepare(self, batch_spec):
        shape = batch_spec.shape
        offset = tuple(
                randint(self.bounding_box[d].start, self.bounding_box[d].stop - shape[d])
                for d in range(len(shape))
        )
        batch_spec.offset = offset

    def process(self, batch):
        pass
