import numpy as np
from batch_filter import BatchFilter

class IntensityAugment(BatchFilter):

    def __init__(self, scale_min, scale_max, shift_min, shift_max, z_section_wise=False):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.shift_min = shift_min
        self.shift_max = shift_max
        self.z_section_wise = z_section_wise

    def process(self, batch):

        assert not self.z_section_wise or len(batch.spec.shape) == 3, "If you specify 'z_section_wise', I expect 3D data."
        assert batch.raw.dtype == np.float32 or batch.raw.dtype == np.float64, "Intensity augmentation requires float types for the raw volume (not " + str(batch.raw.dtype) + "). Consider using Normalize before."
        assert batch.raw.min() >= 0 and batch.raw.max() <= 1, "Intensity augmentation expects raw values in [0,1]. Consider using Normalize before."

        if self.z_section_wise:
            for z in range(batch.spec.shape[0]):
                batch.raw[z] = self.__augment(
                        batch.raw[z],
                        np.random.uniform(low=self.scale_min, high=self.scale_max),
                        np.random.uniform(low=self.shift_min, high=self.shift_max))
        else:
            batch.raw = self.__augment(
                    batch.raw,
                    np.random.uniform(low=self.scale_min, high=self.scale_max),
                    np.random.uniform(low=self.shift_min, high=self.shift_max))

        # clip values, we might have pushed them out of [0,1]
        batch.raw[batch.raw>1] = 1
        batch.raw[batch.raw<0] = 0

    def __augment(self, a, scale, shift):

        return a.mean() + (a-a.mean())*scale + shift
