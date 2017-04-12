import random
import numpy as np
from batch_filter import BatchFilter

class DefectAugment(BatchFilter):

    def __init__(self, prob_missing=0.05, prob_low_contrast=0.05, contrast_scale=0.1, axis=0):

        self.prob_missing = prob_missing
        self.prob_low_contrast = prob_low_contrast
        self.contrast_scale = contrast_scale
        self.axis = axis

    def process(self, batch):

        assert len(batch.spec.shape)==3, "DefectAugment works on 3D batches only"

        prob_missing_threshold = self.prob_missing
        prob_low_contrast_threshold = prob_missing_threshold + self.prob_low_contrast

        for c in range(batch.spec.shape[self.axis]):

            r = random.random()
            print("DefectAugment: r = " + str(r))

            section_selector = tuple(
                    slice(None if d != self.axis else c, None if d != self.axis else c+1)
                    for d in range(len(batch.spec.shape))
            )

            if r < prob_missing_threshold:

                print("Zero-out " + str(section_selector))
                batch.raw[section_selector] = 0

            elif r < prob_low_contrast_threshold:

                section = batch.raw[section_selector].astype(np.float32)

                mean = section.mean()
                section -= mean
                section *= self.contrast_scale
                section += mean

                batch.raw[section_selector] = section
