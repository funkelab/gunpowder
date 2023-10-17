import numpy as np
import random
import skimage

from gunpowder.batch_request import BatchRequest

from .batch_filter import BatchFilter


class NoiseAugment(BatchFilter):
    """Add random noise to an array. Uses the scikit-image function skimage.util.random_noise.
    See scikit-image documentation for more information on arguments and additional kwargs.

    Args:

        array (:class:`ArrayKey`):

            The intensity array to modify. Should be of type float and within range [-1, 1] or [0, 1].

        mode (``string``):

            Type of noise to add, see scikit-image documentation.

        clip (``bool``):

            Whether to preserve the image range (either [-1, 1] or [0, 1]) by clipping values in the end, see
            scikit-image documentation

        p (``float``, optional):

            Probability applying the augmentation. Default is 1.0 (always
            apply). Should be a float value between 0 and 1. Lowering this value
            could be useful for computational efficiency and increasing
            augmentation space.
    """

    def __init__(self, array, mode="gaussian", clip=True, p=1.0, **kwargs):
        self.array = array
        self.mode = mode
        self.clip = clip
        self.p = p
        self.kwargs = kwargs

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def skip_node(self, request):
        return random.random() > self.p

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):
        raw = batch.arrays[self.array]

        assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, (
            "Noise augmentation requires float types for the raw array (not "
            + str(raw.data.dtype)
            + "). Consider using Normalize before."
        )
        if self.clip:
            assert (
                raw.data.min() >= -1 and raw.data.max() <= 1
            ), "Noise augmentation expects raw values in [-1,1] or [0,1]. Consider using Normalize before."

        seed = request.random_seed

        try:
            raw.data = skimage.util.random_noise(
                raw.data, mode=self.mode, rng=seed, clip=self.clip, **self.kwargs
            ).astype(raw.data.dtype)

        except ValueError:
            # legacy version of skimage random_noise
            raw.data = skimage.util.random_noise(
                raw.data, mode=self.mode, seed=seed, clip=self.clip, **self.kwargs
            ).astype(raw.data.dtype)
