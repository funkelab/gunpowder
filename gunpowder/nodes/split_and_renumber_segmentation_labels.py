from .batch_filter import BatchFilter
from gunpowder.ext import malis
from gunpowder.volume import VolumeType

class SplitAndRenumberSegmentationLabels(BatchFilter):

    def process(self, batch, request):
        components = batch.volumes[VolumeType.GT_LABELS].data
        dtype = components.dtype
        simple_neighborhood = malis.mknhood3d()
        affinities_from_components = malis.seg_to_affgraph(
            components,
            simple_neighborhood)
        components, _ = malis.connected_components_affgraph(
            affinities_from_components,
            simple_neighborhood)
        batch.volumes[VolumeType.GT_LABELS].data = components.astype(dtype)
