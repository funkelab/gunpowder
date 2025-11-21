import itertools
import random
import warnings

import numpy as np

from gunpowder.batch_request import BatchRequest

from .batch_filter import BatchFilter


class IntensityAugment(BatchFilter):
    """Randomly scale and shift the values of an intensity array.

    Args:

        array (:class:`ArrayKey`):

            The intensity array to modify.

        scale_min (``float``):
        scale_max (``float``):
        shift_min (``float``):
        shift_max (``float``):

            The min and max of the uniformly randomly drawn scaling and
            shifting values for the intensity augmentation. Intensities are
            changed as::

                a = a.mean() + (a-a.mean())*scale + shift

        z_section_wise (``bool``, optional):

            **Deprecated.** Use ``slab`` instead. This parameter will be removed
            in a future version.

            Perform the augmentation z-section wise. Requires 3D arrays and
            assumes that z is the first dimension.

        clip (``bool``):

            Set to False if modified values should not be clipped to [0, 1]
            Disables range check!

        p (``float``, optional):

            Probability applying the augmentation. Default is 1.0 (always
            apply). Should be a float value between 0 and 1. Lowering this value
            could be useful for computational efficiency and increasing
            augmentation space.

        slab (``tuple`` of ``int``, optional):

            A shape specification to perform the intensity augment in slabs of this
            size. -1 can be used to refer to the actual size of the label
            array. For example, a slab of::

                (2, -1, -1, -1)

            will perform the intensity augment for every each slice ``[0:2,:]``,
            ``[2:4,:]``, ... individually on 4D data.
    """

    def __init__(
        self,
        array,
        scale_min,
        scale_max,
        shift_min,
        shift_max,
        z_section_wise=None,
        clip=True,
        p=1.0,
        slab=None,
    ):
        self.array = array
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.shift_min = shift_min
        self.shift_max = shift_max
        self.z_section_wise = z_section_wise
        if self.z_section_wise is not None:
            warnings.warn(
                DeprecationWarning(
                    "z_section_wise is deprecated and will be removed in the future. Please use slab instead."
                )
            )
            assert slab is None, "z_section_wise and slab are mutually exclusive."

        self.clip = clip
        self.p = p
        self.slab = slab

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
            "Intensity augmentation requires float types for the raw array (not "
            + str(raw.data.dtype)
            + "). Consider using Normalize before."
        )
        if self.clip:
            assert (
                raw.data.min() >= 0 and raw.data.max() <= 1
            ), "Intensity augmentation expects raw values in [0,1]. Consider using Normalize before."

        if self.z_section_wise:
            slab = [-1] * len(raw.data.shape)
            slab[-raw.spec.voxel_size.dims] = 1
        elif self.slab is not None:
            slab = self.slab
        else:
            slab = [-1] * len(raw.data.shape)

        # slab with -1 replaced by shape
        slab = tuple(m if s == -1 else s for m, s in zip(raw.data.shape, slab))

        slab_ranges = (range(0, m, s) for m, s in zip(raw.data.shape, slab))

        for start in itertools.product(*slab_ranges):
            slices = tuple(
                slice(start[d], start[d] + slab[d]) for d in range(len(slab))
            )
            raw.data[slices] = self.__augment(
                raw.data[slices],
                np.random.uniform(low=self.scale_min, high=self.scale_max),
                np.random.uniform(low=self.shift_min, high=self.shift_max),
            )

        # clip values, we might have pushed them out of [0,1]
        if self.clip:
            raw.data[raw.data > 1] = 1
            raw.data[raw.data < 0] = 0

    def __augment(self, a, scale, shift):
        return a.mean() + (a - a.mean()) * scale + shift
