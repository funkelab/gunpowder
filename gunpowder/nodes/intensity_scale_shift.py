from .batch_filter import BatchFilter

class IntensityScaleShift(BatchFilter):
    '''Scales the intensities of a batch by 'scale', then adds 'shift'.

    This is useful to transform your intensities into the interval [-1,1], as is 
    needed before passing them to the CNN.
    '''

    def __init__(self, intensities, scale, shift):
        self.intensities = intensities
        self.scale = scale
        self.shift = shift

    def process(self, batch, request):

        raw = batch.volumes[self.intensities]
        raw.data = raw.data*self.scale + self.shift
