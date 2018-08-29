from .batch_filter import BatchFilter
from gunpowder.array import Array
import collections
import itertools
import logging
import numpy as np

logger = logging.getLogger(__name__)


class BalanceLabels(BatchFilter):
    '''Creates a scale array to balance the loss between class labels.

    Note that this only balances loss weights per-batch and does not accumulate
    statistics about class balance across batches.

    Args:

        labels (:class:`ArrayKey`):

            An array containing binary or integer labels.

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

        num_classes(``int``, optional):

            The number of classes. Labels will be expected to be in the
            interval [0, ``num_classes``). Defaults to 2 for binary
            classification.
    '''

    def __init__(self, labels, scales, mask=None, slab=None, num_classes=2):

        self.labels = labels
        self.scales = scales
        if mask is None:
            self.masks = []
        elif not isinstance(mask, collections.Iterable):
            self.masks = [mask]
        else:
            self.masks = mask

        self.slab = slab
        self.num_classes = num_classes

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

        assert len(np.unique(labels.data)) <= self.num_classes, (
            "Found more unique labels than classes in %s."%self.labels)
        assert 0 <= np.min(labels.data) < self.num_classes, (
            "Labels %s are not in [0, num_classes)."%self.labels)
        assert 0 <= np.max(labels.data) < self.num_classes, (
            "Labels %s are not in [0, num_classes)."%self.labels)

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

        # in the masked-in area, compute the fraction of per-class samples
        masked_in = scale.sum()
        classes, counts = np.unique(labels[np.nonzero(scale)], return_counts=True)
        fracs = counts.astype(float) / masked_in if masked_in > 0 else np.zeros(counts.size)
        np.clip(fracs, 0.05, 0.95, fracs)

        # compute the class weights
        w_sparse = 1.0 / float(self.num_classes) / fracs
        w = np.zeros(self.num_classes)
        w[classes] = w_sparse

        # scale the masked-in scale with the class weights
        scale *= np.take(w, labels)
