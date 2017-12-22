from .batch_filter import BatchFilter
from gunpowder.ext import malis

class SplitAndRenumberSegmentationLabels(BatchFilter):

    def __init__(self, labels):
        self.labels = labels

    def process(self, batch, request):
        components = batch.volumes[self.labels].data
        dtype = components.dtype
        simple_neighborhood = malis.mknhood3d()
        affinities_from_components = malis.seg_to_affgraph(
            components,
            simple_neighborhood)
        components, _ = malis.connected_components_affgraph(
            affinities_from_components,
            simple_neighborhood)
        batch.volumes[self.labels].data = components.astype(dtype)
