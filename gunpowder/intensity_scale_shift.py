from batch_filter import BatchFilter

class IntensityScaleShift(BatchFilter):
    '''Scales the intensities of a batch by 'scale', then adds 'shift'.

    This is useful to transform your intensities into the interval [-1,1], as is 
    needed before passing them to the CNN.
    '''

    def __init__(self, scale, shift):
        self.scale = scale
        self.shift = shift

    def process(self, batch):

        batch.raw = batch.raw*self.scale + self.shift
