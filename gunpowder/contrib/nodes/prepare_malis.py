import copy
import logging
import numpy as np

from gunpowder.volume import Array
from gunpowder.nodes.batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class PrepareMalis(BatchFilter):
    ''' Creates a component label volume needed for two-phase malis training.

    Args:

        labels_volume_type(:class:`ArrayType`): The label volume to use.

        malis_comp_volume_type(:class:`ArrayType`): The malis component volume
            to generate.

        ignore_volume_type(:class:`ArrayType`, optional): An ignore mask to
            use.
    '''

    def __init__(
            self,
            labels_volume_type,
            malis_comp_volume_type,
            ignore_volume_type=None):

        self.labels_volume_type = labels_volume_type
        self.malis_comp_volume_type = malis_comp_volume_type
        self.ignore_volume_type = ignore_volume_type

    def setup(self):

        spec = self.spec[self.labels_volume_type].copy()
        self.provides(self.malis_comp_volume_type, spec)
        self.enable_autoskip()

    def prepare(self, request):

        assert self.labels_volume_type in request, (
            "PrepareMalis requires %s, but they are not in request"%
            self.labels_volume_type)

    def process(self, batch, request):

        gt_labels = batch.volumes[self.labels_volume_type]
        next_id = gt_labels.data.max() + 1

        gt_pos_pass = gt_labels.data

        if self.ignore_volume_type and self.ignore_volume_type in batch.volumes:

            gt_neg_pass = np.array(gt_labels.data)
            gt_neg_pass[
                batch.volumes[self.ignore_volume_type].data == 0] = next_id

        else:

            gt_neg_pass = gt_pos_pass

        spec = self.spec[self.malis_comp_volume_type].copy()
        spec.roi = request[self.labels_volume_type].roi
        batch.volumes[self.malis_comp_volume_type] = Array(
            np.array([gt_neg_pass, gt_pos_pass]),
            spec)

        # Why don't we update gt_affinities in the same way?
        # -> not needed
        #
        # GT affinities are all 0 in the masked area (because masked area is
        # assumed to be set to background in batch.gt).
        #
        # In the negative pass:
        #
        #   We set all affinities inside GT regions to 1. Affinities in masked
        #   area as predicted. Belongs to one forground region (introduced
        #   above). But we only count loss on edges connecting different labels
        #   -> loss in masked-out area only from outside regions.
        #
        # In the positive pass:
        #
        #   We set all affinities outside GT regions to 0 -> no loss in masked
        #   out area.
