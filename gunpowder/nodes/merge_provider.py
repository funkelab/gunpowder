from gunpowder.provider_spec import ProviderSpec
from gunpowder.batch import Batch
from gunpowder.batch_request import BatchRequest

from .batch_provider import BatchProvider


class MergeProvider(BatchProvider):
    '''Merges different providers.
    '''
    def __init__(self):
        self.identifier_to_provider = {}

    def setup(self):
        assert len(self.get_upstream_providers()) > 0, "at least one batch provider needs to be added to the MergeProvider"
        # Only allow merging if no two upstream_providers have the same volume/points_type
        error_message = "Type {} provided by more than one upstream provider. Node MergeProvider only allows to merge " \
                        "providers with different types."
        for provider in self.get_upstream_providers():
            for identifier, spec in provider.spec.items():
                assert self.spec is None or identifier not in self.spec, error_message.format(identifier)
                self.provides(identifier, spec)
                self.identifier_to_provider[identifier] = provider

    def provide(self, request):

        # create upstream requests
        upstream_requests = {}
        for identifier, spec in request.items():

            provider = self.identifier_to_provider[identifier]
            if provider not in upstream_requests:
                upstream_requests[provider] = BatchRequest()

            upstream_requests[provider][identifier] = spec

        # execute requests, merge batches
        merged_batch = Batch()
        for provider, upstream_request in upstream_requests.items():

            batch = provider.request_batch(upstream_request)
            for identifier, volume in batch.volumes.items():
                merged_batch.volumes[identifier] = volume
            for identifier, points in batch.points.items():
                merged_batch.points[identifier] = points

        return merged_batch


