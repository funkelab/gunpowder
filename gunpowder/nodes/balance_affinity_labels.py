from .batch_filter import BatchFilter
from gunpowder.volume import VolumeTypes, Volume
import logging
import numpy as np

logger = logging.getLogger(__name__)

class BalanceAffinityLabels(BatchFilter):
    '''Creates a LOSS_SCALE volume that balances the loss between positive and 
    negative affinities.
    '''

    def __init__(self):
        self.skip_next = False

    def prepare(self, request):

        if VolumeTypes.LOSS_SCALE not in request.volumes:
            self.skip_next = True
        else:
            del request.volumes[VolumeTypes.LOSS_SCALE]

    def process(self, batch, request):

        if self.skip_next:
            self.skip_next = False
            return

        gt_affinities = batch.volumes[VolumeTypes.GT_AFFINITIES]

        # initialize error scale with 1s
        error_scale = np.ones(gt_affinities.data.shape, dtype=np.float)

        # set error_scale to 0 in masked-out areas
        if VolumeTypes.GT_MASK in batch.volumes:
            self.__mask_error_scale(error_scale, batch.volumes[VolumeTypes.GT_MASK].data)
        if VolumeTypes.GT_IGNORE in batch.volumes:
            self.__mask_error_scale(error_scale, batch.volumes[VolumeTypes.GT_IGNORE].data)

        # in the masked-in area, compute the fraction of positive samples
        masked_in = error_scale.sum()
        num_pos = (gt_affinities.data*error_scale).sum()
        frac_pos = float(num_pos)/masked_in if masked_in > 0 else 0
        frac_pos = np.clip(frac_pos, 0.05, 0.95)
        frac_neg = 1.0 - frac_pos

        # compute the class weights for positive and negative samples
        w_pos = 1.0/(2.0*frac_pos)
        w_neg = 1.0/(2.0*frac_neg)

        # scale the masked-in error_scale with the class weights
        error_scale *= (gt_affinities.data >= 0.5)*w_pos + (gt_affinities.data < 0.5)*w_neg

        batch.volumes[VolumeTypes.LOSS_SCALE] = Volume(
                error_scale,
                gt_affinities.roi,
                gt_affinities.resolution)

    def __mask_error_scale(self, error_scale, mask):
        for d in range(error_scale.shape[0]):
            error_scale[d] *= mask
