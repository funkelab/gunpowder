from .batch_filter import BatchFilter
from gunpowder.volume import VolumeTypes, Volume
import copy
import collections
import logging
import numpy as np

logger = logging.getLogger(__name__)


class BalanceLabels(BatchFilter):
    '''Creates loss_scale_volumes which apply the provided mask_volumes and balance the loss between positive
        and negative labels in the label_volume.
        The labels are balanced after the mask_volumes are applied to their corresponding label_volume.
    Args:
        labels_to_loss_scale_volume (dict): Dictionary from :class:``VolumeType`` of the labels to be scaled (label_volume)
                                        to the :class:``VolumeTypes`` which will be created to store the loss scale volume
                                        (loss_scale_volume). This new volume will have the same ROI and resolution
                                        as the label_volume.
        labels_to_mask_volumes (dict): Dictionary from :class:``VolumeType`` of the labels (label_volume) to be scaled
                                        to a list or tuple of :class:``VolumeTypes`` of the masks to be applied to this
                                        label_volume.
    '''

    def __init__(self, labels_to_loss_scale_volume, labels_to_mask_volumes=None):
        self.labels_to_loss_scale_volume = labels_to_loss_scale_volume
        if labels_to_mask_volumes is None:
            self.labels_to_mask_volumes = []
        else:
            self.labels_to_mask_volumes = labels_to_mask_volumes
            for label_volume, mask_volumes in self.labels_to_mask_volumes.items():
                if not isinstance(mask_volumes, collections.Iterable):
                    self.labels_to_mask_volumes[label_volume] = [mask_volumes]

        self.skip_next = False

    def setup(self):

        for (label_volume, loss_scale_volume) in self.labels_to_loss_scale_volume.items():

            assert label_volume in self.spec, "Asked to balance labels {}, which are not provided.".format(label_volume)

            if label_volume in self.labels_to_mask_volumes:
                for mask_volume in self.labels_to_mask_volumes[label_volume]:
                    assert mask_volume in self.spec, "Asked to apply mask {} to balance labels, but mask is not provided.".format(mask_volume)

            spec = copy.deepcopy(self.spec[label_volume])
            spec.dtype = np.float32
            self.provides(loss_scale_volume, spec)

    def prepare(self, request):

        self.skip_next = True
        for _, loss_scale_volume in self.labels_to_loss_scale_volume.items():
            if loss_scale_volume in request:
                del request[loss_scale_volume]
                self.skip_next = False

    def process(self, batch, request):

        if self.skip_next:
            self.skip_next = False
            return

        for label_volume, loss_scale_volume in self.labels_to_loss_scale_volume.items():

            labels = batch.volumes[label_volume]

            # initialize error scale with 1s
            error_scale = np.ones(labels.data.shape, dtype=np.float32)

            # set error_scale to 0 in masked-out areas
            if label_volume in self.labels_to_mask_volumes:
                for mask_volume in self.labels_to_mask_volumes[label_volume]:
                    self.__mask_error_scale(error_scale, batch.volumes[mask_volume].data)

            # in the masked-in area, compute the fraction of positive samples
            masked_in = error_scale.sum()
            labels_binary = np.floor(np.clip(labels.data+0.5, a_min=0, a_max=1))
            num_pos  = (labels_binary * error_scale).sum()
            frac_pos = float(num_pos) / masked_in if masked_in > 0 else 0
            frac_pos = np.clip(frac_pos, 0.05, 0.95)
            frac_neg = 1.0 - frac_pos

            # compute the class weights for positive and negative samples
            w_pos = 1.0 / (2.0 * frac_pos)
            w_neg = 1.0 / (2.0 * frac_neg)

            # scale the masked-in error_scale with the class weights
            error_scale *= (labels_binary >= 0.5) * w_pos + (labels_binary < 0.5) * w_neg

            spec = copy.deepcopy(self.spec[loss_scale_volume])
            spec.roi = copy.deepcopy(labels.spec.roi)
            batch.volumes[loss_scale_volume] = Volume(error_scale, spec)

    def __mask_error_scale(self, error_scale, mask):

        if error_scale.shape == mask.shape:
            error_scale = error_scale[np.newaxis, :]
        for d in range(error_scale.shape[0]):
            error_scale[d] *= mask
