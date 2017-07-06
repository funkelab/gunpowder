import copy
import random

from .batch_provider import BatchProvider

class RandomProvider(BatchProvider):
    '''Randomly selects one of the upstream providers.
    '''

    def setup(self):

        self.spec = None
        assert len(self.get_upstream_providers()) > 0, "at least one batch provider needs to be added to the RandomProvider"

        # advertise volume_types only if all upstream providers have them
        for provider in self.get_upstream_providers():
            if self.spec is None:
                self.spec = copy.deepcopy(provider.get_spec())
            else:
                my_volume_types = list(self.spec.volumes.keys())
                for volume_type in my_volume_types:
                    if volume_type not in provider.get_spec().volumes:
                        del self.spec.volumes[volume_type]

                my_points_types = list(self.spec.points.keys())
                for points_type in my_points_types:
                    if points_type not in provider.get_spec().points:
                        del self.spec.points[points_type]

    def get_spec(self):
        return self.spec

    def provide(self, request):
        return random.choice(self.get_upstream_providers()).request_batch(request)
