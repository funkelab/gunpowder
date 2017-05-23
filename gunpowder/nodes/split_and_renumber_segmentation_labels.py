import malis

from batch_filter import BatchFilter


class SplitAndRenumberSegmentationLabels(BatchFilter):

    def process(self, batch):
        components = batch.gt.copy()
        shape = batch.gt.shape
        dtype = batch.gt.dtype
        simple_neighborhood = malis.mknhood3d()
        affinities_from_components = malis.seg_to_affgraph(
            components.reshape(shape[-3:]),
            simple_neighborhood)
        components, _ = malis.connected_components_affgraph(
            affinities_from_components,
            simple_neighborhood)
        batch.gt[:] = components.reshape(shape).astype(dtype)
