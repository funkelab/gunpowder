from gunpowder.provider_spec import ProviderSpec
from gunpowder.batch import Batch
from gunpowder.batch_request import BatchRequest

from .batch_provider import BatchProvider


class MergeProvider(BatchProvider):
    '''Merges different providers::

        (a + b + c) + MergeProvider()

    will create a provider that combines the arrays and points offered by
    ``a``, ``b``, and ``c``. Array and point keys of ``a``, ``b``, and ``c`` should be
    the disjoint.
    '''
    def __init__(self):
        self.key_to_provider = {}

    def setup(self):
        assert len(self.get_upstream_providers()) > 0, "at least one batch provider needs to be added to the MergeProvider"
        # Only allow merging if no two upstream_providers have the same
        # array/points keys
        error_message = "Key {} provided by more than one upstream provider. Node MergeProvider only allows to merge " \
                        "providers with different keys."
        for provider in self.get_upstream_providers():
            for key, spec in provider.spec.items():
                assert self.spec is None or key not in self.spec, error_message.format(key)
                self.provides(key, spec)
                self.key_to_provider[key] = provider

    def provide(self, request):

        # create upstream requests
        upstream_requests = {}
        for key, spec in request.items():

            provider = self.key_to_provider[key]
            if provider not in upstream_requests:
                upstream_requests[provider] = BatchRequest()

            upstream_requests[provider][key] = spec

        # execute requests, merge batches
        merged_batch = Batch()
        for provider, upstream_request in upstream_requests.items():

            batch = provider.request_batch(upstream_request)
            for key, array in batch.arrays.items():
                merged_batch.arrays[key] = array
            for key, points in batch.points.items():
                merged_batch.points[key] = points

        return merged_batch


