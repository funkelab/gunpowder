from .batch_filter import BatchFilter
from gunpowder.volume import VolumeType

class IntensityScaleShift(BatchFilter):
    '''Scales the intensities of a batch by 'scale', then adds 'shift'.

    This is useful to transform your intensities into the interval [-1,1], as is 
    needed before passing them to the CNN.
    '''

    def __init__(self, scale, shift):
        self.scale = scale
        self.shift = shift

    def process(self, batch, request):

        raw = batch.volumes[VolumeType.RAW]
        raw.data = raw.data*self.scale + self.shift
