from .batch_filter import BatchFilter

class IntensityScaleShift(BatchFilter):
    '''Scales the intensities of a batch by ``scale``, then adds ``shift``.

    Args:

        array (:class:`ArrayKey`):

            The key of the array to modify.

        scale (``float``):
        shift (``float``):

            The shift and scale to apply to ``array``.
    '''

    def __init__(self, array, scale, shift):
        self.array = array
        self.scale = scale
        self.shift = shift

    def process(self, batch, request):

        raw = batch.arrays[self.array]
        raw.data = raw.data*self.scale + self.shift
