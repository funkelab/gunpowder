from gunpowder.provider_spec import ProviderSpec
from gunpowder.batch import Batch
from gunpowder.batch_request import BatchRequest

from .batch_provider import BatchProvider


class MergeProvider(BatchProvider):
    '''Merges different providers.
    '''
    def __init__(self):
        self.volumetype_to_provider = {}
        self.pointstype_to_provider = {}

    def setup(self):
        self.spec = ProviderSpec()
        assert len(self.get_upstream_providers()) > 0, "at least one batch provider needs to be added to the MergeProvider"
        # Only allow merging if no two upstream_providers have the same volume/points_type
        error_message = "Type {} provided by more than one upstream provider. Node MergeProvider only allows to merge " \
                        "providers with different types."
        for provider in self.get_upstream_providers():
            for volume_type, volume_rois in provider.get_spec().volumes.items():
                assert volume_type not in self.spec.volumes, error_message.format(volume_type)
                self.spec.volumes[volume_type] = volume_rois
                self.volumetype_to_provider[volume_type] = provider
            for points_type, points_rois in provider.get_spec().points.items():
                assert points_type not in self.spec.points, error_message.format(points_type)
                self.spec.points[points_type] = points_rois
                self.pointstype_to_provider[points_type] = provider

    def get_spec(self):
        return self.spec

    def provide(self, request):
        fused_batch = Batch()
        for collection_type in [request.volumes, request.points]:
            for type, roi in collection_type.items():
                if type in self.volumetype_to_provider:
                    cur_provider = self.volumetype_to_provider[type]
                    cur_request = BatchRequest(initial_volumes={type: roi})
                    cur_batch = cur_provider.request_batch(cur_request)
                    fused_batch.volumes[type] = cur_batch.volumes[type]
                elif type in self.pointstype_to_provider:
                    cur_provider = self.pointstype_to_provider[type]
                    cur_request = BatchRequest(initial_points={type: roi})
                    cur_batch = cur_provider.request_batch(cur_request)
                    fused_batch.points[type] = cur_batch.points[type]
                else:
                    raise Exception("Requested %s, but none of the sources provides it." % type)
        return fused_batch


