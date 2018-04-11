from .batch_filter import BatchFilter
from gunpowder.array import Array
import collections
import itertools
import logging
import numpy as np

logger = logging.getLogger(__name__)


class BalanceLabels(BatchFilter):
    '''Creates a scale array to balance the loss between positive and negative
    labels.

    Args:

        labels (:class:`ArrayKey`):

            A array containing binary labels.

        scales (:class:`ArrayKey`):

            A array with scales to be created. This new array will have the
            same ROI and resolution as ``labels``.

        mask (:class:`ArrayKey`, optional):

            An optional mask (or list of masks) to consider for balancing.
            Every voxel marked with a 0 will not contribute to the scaling and
            will have a scale of 0 in ``scales``.

        slab (``tuple`` of ``int``, optional):

            A shape specification to perform the balancing in slabs of this
            size. -1 can be used to refer to the actual size of the label
            array. For example, a slab of::

                (2, -1, -1, -1)

            will perform the balancing for every each slice ``[0:2,:]``,
            ``[2:4,:]``, ... individually.
    '''

    def __init__(self, labels, scales, mask=None, slab=None):

        self.labels = labels
        self.scales = scales
        if mask is None:
            self.masks = []
        elif not isinstance(mask, collections.Iterable):
            self.masks = [mask]
        else:
            self.masks = mask

        self.slab = slab

    def setup(self):

        assert self.labels in self.spec, (
            "Asked to balance labels %s, which are not provided."%self.labels)

        for mask in self.masks:
            assert mask in self.spec, (
                "Asked to apply mask %s to balance labels, but mask is not "
                "provided."%mask)

        spec = self.spec[self.labels].copy()
        spec.dtype = np.float32
        self.provides(self.scales, spec)
        self.enable_autoskip()

    def process(self, batch, request):

        labels = batch.arrays[self.labels]

        assert len(np.unique(labels.data)) <= 2, (
            "Found more than two labels in %s."%self.labels)
        assert np.min(labels.data) in [0.0, 1.0], (
            "Labels %s are not binary."%self.labels)
        assert np.max(labels.data) in [0.0, 1.0], (
            "Labels %s are not binary."%self.labels)

        # initialize error scale with 1s
        error_scale = np.ones(labels.data.shape, dtype=np.float32)

        # set error_scale to 0 in masked-out areas
        for key in self.masks:
            mask = batch.arrays[key]
            assert labels.data.shape == mask.data.shape, (
                "Shape of mask %s %s does not match %s %s"%(
                    mask,
                    mask.data.shape,
                    self.labels,
                    labels.data.shape))
            error_scale *= mask.data

        if not self.slab:
            slab = error_scale.shape
        else:
            # slab with -1 replaced by shape
            slab = tuple(
                m if s == -1 else s
                for m, s in zip(error_scale.shape, self.slab))

        slab_ranges = (
            range(0, m, s)
            for m, s in zip(error_scale.shape, slab))

        for start in itertools.product(*slab_ranges):
            slices = tuple(
                slice(start[d], start[d] + slab[d])
                for d in range(len(slab)))
            self.__balance(
                labels.data[slices],
                error_scale[slices])

        spec = self.spec[self.scales].copy()
        spec.roi = labels.spec.roi
        batch.arrays[self.scales] = Array(error_scale, spec)

    def __balance(self, labels, scale):

        # in the masked-in area, compute the fraction of positive samples
        masked_in = scale.sum()
        num_pos  = (labels*scale).sum()
        frac_pos = float(num_pos) / masked_in if masked_in > 0 else 0
        frac_pos = np.clip(frac_pos, 0.05, 0.95)
        frac_neg = 1.0 - frac_pos

        # compute the class weights for positive and negative samples
        w_pos = 1.0 / (2.0 * frac_pos)
        w_neg = 1.0 / (2.0 * frac_neg)

        # scale the masked-in scale with the class weights
        scale *= (labels >= 0.5) * w_pos + (labels < 0.5) * w_neg
