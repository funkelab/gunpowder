import copy
import logging
import numpy as np

from .batch_filter import BatchFilter
from gunpowder.volume import Volume, VolumeTypes

logger = logging.getLogger(__name__)

class PrepareMalis(BatchFilter):
    ''' Creates volume MALIS_COMP_LABEL for malis training '''

    def __init__(self):
        self.skip_next = False

    def setup(self):
        self.upstream_spec = self.get_upstream_provider().get_spec()
        self.spec = copy.deepcopy(self.upstream_spec)

        # only give warning that GT_LABELS is missing, in prepare() when checked that node is not skipped
        if VolumeTypes.GT_LABELS in self.spec.volumes:
            self.spec.volumes[VolumeTypes.MALIS_COMP_LABEL] = self.spec.volumes[VolumeTypes.GT_LABELS]

    def get_spec(self):
        return self.spec

    def prepare(self, request):

        # MALIS_COMP_LABEL has to be in request for PrepareMalis to run
        if not VolumeTypes.MALIS_COMP_LABEL in request.volumes:
            logger.warn("no {} requested, will do nothing".format(VolumeTypes.MALIS_COMP_LABEL))
            self.skip_next = True
        else:
            assert VolumeTypes.GT_LABELS in request.volumes, "PrepareMalis requires GT_LABELS, but they are not in request"
            del request.volumes[VolumeTypes.MALIS_COMP_LABEL]

    def process(self, batch, request):

        # do nothing if MALIS_COMP_LABEL is not in request
        if self.skip_next:
            self.skip_next = False
            return

        gt_labels = batch.volumes[VolumeTypes.GT_LABELS]
        next_id = gt_labels.data.max() + 1

        gt_pos_pass = gt_labels.data

        if VolumeTypes.GT_IGNORE in batch.volumes:

            gt_neg_pass = np.array(gt_labels.data)
            gt_neg_pass[batch.volumes[VolumeTypes.GT_IGNORE].data==0] = next_id

        else:

            gt_neg_pass = gt_pos_pass

        batch.volumes[VolumeTypes.MALIS_COMP_LABEL] = Volume(data=np.array([gt_neg_pass, gt_pos_pass]),
                                                             roi=request.volumes[VolumeTypes.GT_LABELS],
                                                             resolution=batch.volumes[VolumeTypes.GT_LABELS].resolution)

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
