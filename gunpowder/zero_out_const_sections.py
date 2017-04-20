from batch_filter import BatchFilter

class ZeroOutConstSections(BatchFilter):
    '''Every z-section that has constant values only will be set to 0.

    This is to handle blank (missing) sections in a less invasive way: Instead 
    of leaving them at -1 (which is "black", the lowest possible input to the 
    CNN), 0 ("gray") might be easier to ignore.

    For that you should call this filter after you are done with all other 
    intensity manipulations.
    '''

    def process(self, batch):

        assert batch.spec.input_roi.dims() == 3, "This filter only works on 3D data."

        for z in range(batch.spec.input_roi.get_shape()[0]):
            if batch.raw[z].min() == batch.raw[z].max():
                batch.raw[z] = 0
