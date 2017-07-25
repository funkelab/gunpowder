import logging
import numpy as np
import random

from .batch_filter import BatchFilter
from gunpowder.batch_request import BatchRequest
from gunpowder.build import build
from gunpowder.coordinate import Coordinate
from gunpowder.volume import VolumeTypes

logger = logging.getLogger(__name__)

class DefectAugment(BatchFilter):

    def __init__(
            self,
            prob_missing=0.05,
            prob_low_contrast=0.05,
            prob_artifact=0.0,
            contrast_scale=0.1,
            artifact_source=None,
            axis=0):
        '''Create a new DefectAugment node.

        Args

            prob_missing, prob_low_contrast, prob_artifact:

                Probabilities of having a missing section, low-contrast section, 
                or an artifact (see param 'artifact_source'). The sum should not 
                exceed 1.

            contrast_scale:

                By how much to scale the intensities for a low-contrast section.

            artifact_source:

                A gunpowder batch provider that delivers VolumeTypes.RAW and 
                VolumeTypes.ALPHA_MASK, used if prob_artifact > 0.

            axis:

                Along which axis sections a cut.
        '''
        self.prob_missing = prob_missing
        self.prob_low_contrast = prob_low_contrast
        self.prob_artifact = prob_artifact
        self.contrast_scale = contrast_scale
        self.artifact_source = artifact_source
        self.axis = axis

    def setup(self):

        if self.artifact_source is not None:
            self.artifact_source.setup()

    def teardown(self):

        if self.artifact_source is not None:
            self.artifact_source.teardown()

    def process(self, batch, request):

        assert batch.get_total_roi().dims()==3, "DefectAugment works on 3D batches only"

        prob_missing_threshold = self.prob_missing
        prob_low_contrast_threshold = prob_missing_threshold + self.prob_low_contrast
        prob_artifact_threshold = prob_low_contrast_threshold + self.prob_artifact

        raw = batch.volumes[VolumeTypes.RAW]

        for c in range((raw.roi/VolumeTypes.RAW.voxel_size).get_shape()[self.axis]):

            r = random.random()

            section_selector = tuple(
                    slice(None if d != self.axis else c, None if d != self.axis else c+1)
                    for d in range(raw.roi.dims())
            )

            if r < prob_missing_threshold:

                logger.debug("Zero-out " + str(section_selector))
                raw.data[section_selector] = 0

            elif r < prob_low_contrast_threshold:

                logger.debug("Lower contrast " + str(section_selector))
                section = raw.data[section_selector]

                mean = section.mean()
                section -= mean
                section *= self.contrast_scale
                section += mean

                raw.data[section_selector] = section

            elif r < prob_artifact_threshold:

                logger.debug("Add artifact " + str(section_selector))
                section = raw.data[section_selector]

                assert VolumeTypes.RAW.voxel_size == VolumeTypes.ALPHA_MASK.voxel_size, \
                        "Can only alpha blend RAW with ALPHA_MASK if both have the same voxel size"

                artifact_request = BatchRequest()
                artifact_request.add_volume_request(VolumeTypes.RAW, Coordinate(section.shape)*VolumeTypes.RAW.voxel_size)
                artifact_request.add_volume_request(VolumeTypes.ALPHA_MASK, Coordinate(section.shape)*VolumeTypes.ALPHA_MASK.voxel_size)
                logger.debug("Requesting artifact batch " + str(artifact_request))

                artifact_batch = self.artifact_source.request_batch(artifact_request)
                artifact_alpha = artifact_batch.volumes[VolumeTypes.ALPHA_MASK].data
                artifact_raw   = artifact_batch.volumes[VolumeTypes.RAW].data

                assert artifact_raw.dtype == section.dtype
                assert artifact_alpha.dtype == np.float32
                assert artifact_alpha.min() >= 0.0
                assert artifact_alpha.max() <= 1.0

                raw.data[section_selector] = section*(1.0 - artifact_alpha) + artifact_raw*artifact_alpha
