import copy
import logging
import numpy as np

from gunpowder.array import Array
from gunpowder.nodes.batch_filter import BatchFilter

logger = logging.getLogger(__name__)

class PrepareMalis(BatchFilter):
    ''' Creates a component label array needed for two-phase malis training.

    Args:

        labels_array_key(:class:`ArrayKey`): The label array to use.

        malis_comp_array_key(:class:`ArrayKey`): The malis component array
            to generate.

        ignore_array_key(:class:`ArrayKey`, optional): An ignore mask to
            use.
    '''

    def __init__(
            self,
            labels_array_key,
            malis_comp_array_key,
            ignore_array_key=None):

        self.labels_array_key = labels_array_key
        self.malis_comp_array_key = malis_comp_array_key
        self.ignore_array_key = ignore_array_key

    def setup(self):

        spec = self.spec[self.labels_array_key].copy()
        self.provides(self.malis_comp_array_key, spec)
        self.enable_autoskip()

    def prepare(self, request):

        assert self.labels_array_key in request, (
            "PrepareMalis requires %s, but they are not in request"%
            self.labels_array_key)

    def process(self, batch, request):

        gt_labels = batch.arrays[self.labels_array_key]
        next_id = gt_labels.data.max() + 1

        gt_pos_pass = gt_labels.data

        if self.ignore_array_key and self.ignore_array_key in batch.arrays:

            gt_neg_pass = np.array(gt_labels.data)
            gt_neg_pass[
                batch.arrays[self.ignore_array_key].data == 0] = next_id

        else:

            gt_neg_pass = gt_pos_pass

        spec = self.spec[self.malis_comp_array_key].copy()
        spec.roi = request[self.labels_array_key].roi
        batch.arrays[self.malis_comp_array_key] = Array(
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
