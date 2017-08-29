from .batch_filter import BatchFilter
from gunpowder.volume import VolumeTypes

class ZeroOutConstSections(BatchFilter):
    '''Every z-section that has constant values only will be set to 0.

    This is to handle blank (missing) sections in a less invasive way: Instead 
    of leaving them at -1 (which is "black", the lowest possible input to the 
    CNN), 0 ("gray") might be easier to ignore.

    For that you should call this filter after you are done with all other 
    intensity manipulations.
    '''

    def process(self, batch, request):

        assert batch.get_total_roi().dims() == 3, "This filter only works on 3D data."

        raw = batch.volumes[VolumeTypes.RAW]

        for z in range((raw.spec.roi/self.spec[VolumeTypes.RAW].voxel_size).get_shape()[0]):
            if raw.data[z].min() == raw.data[z].max():
                raw.data[z] = 0
