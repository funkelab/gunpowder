import copy
import random

from .batch_provider import BatchProvider

class RandomProvider(BatchProvider):
    '''Randomly selects one of the upstream providers.
    '''

    def setup(self):

        assert len(self.get_upstream_providers()) > 0, "at least one batch provider needs to be added to the RandomProvider"

        common_spec = None

        # advertise volume_types only if all upstream providers have them
        for provider in self.get_upstream_providers():

            if common_spec is None:
                common_spec = copy.deepcopy(provider.get_spec())
            else:
                my_volume_types = list(common_spec.volume_specs.keys())
                for volume_type in my_volume_types:
                    if volume_type not in provider.get_spec().volume_specs:
                        del common_spec.volume_specs[volume_type]

                my_points_types = list(common_spec.points_specs.keys())
                for points_type in my_points_types:
                    if points_type not in provider.get_spec().points_specs:
                        del common_spec.points_specsc[points_type]

        for volume_type, spec in common_spec.volume_specs.items():
            self.add_volume_spec(volume_type, spec)
        for points_type, spec in common_spec.points_specs.items():
            self.add_points_spec(points_type, spec)

    def provide(self, request):
        return random.choice(self.get_upstream_providers()).request_batch(request)
