import numpy as np
import skimage

from .batch_filter import BatchFilter

class NoiseAugment(BatchFilter):
    '''Add random noise to an array. Uses the scikit-image function skimage.util.random_noise.
    See scikit-image documentation for more information on arguments and additional kwargs.

    Args:

        array (:class:`ArrayKey`):

            The intensity array to modify. Should be of type float and within range [-1, 1] or [0, 1].

        mode (``string``):

            Type of noise to add, see scikit-image documentation.

        seed (``int``):

            Optionally set a random seed, see scikit-image documentation.

        clip (``bool``):

            Whether to preserve the image range (either [-1, 1] or [0, 1]) by clipping values in the end, see
            scikit-image documentation
    '''

    def __init__(self, array, mode='gaussian', seed=None, clip=True, **kwargs):
        self.array = array
        self.mode = mode
        self.seed = seed
        self.clip = clip
        self.kwargs = kwargs

    def process(self, batch, request):

        raw = batch.arrays[self.array]

        assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, "Noise augmentation requires float types for the raw array (not " + str(raw.data.dtype) + "). Consider using Normalize before."
        assert raw.data.min() >= -1 and raw.data.max() <= 1, "Noise augmentation expects raw values in [-1,1] or [0,1]. Consider using Normalize before."
        raw.data = skimage.util.random_noise(raw.data, mode=self.mode, seed=self.seed, clip=self.clip, **self.kwargs )
