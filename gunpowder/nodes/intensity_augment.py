import numpy as np

from .batch_filter import BatchFilter
from gunpowder.volume import VolumeTypes

class IntensityAugment(BatchFilter):

    def __init__(self, scale_min, scale_max, shift_min, shift_max, z_section_wise=False):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.shift_min = shift_min
        self.shift_max = shift_max
        self.z_section_wise = z_section_wise

    def process(self, batch, request):

        raw = batch.volumes[VolumeTypes.RAW]

        assert not self.z_section_wise or raw.spec.roi.dims() == 3, "If you specify 'z_section_wise', I expect 3D data."
        assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, "Intensity augmentation requires float types for the raw volume (not " + str(raw.data.dtype) + "). Consider using Normalize before."
        assert raw.data.min() >= 0 and raw.data.max() <= 1, "Intensity augmentation expects raw values in [0,1]. Consider using Normalize before."

        if self.z_section_wise:
            for z in range((raw.spec.roi/self.spec[VolumeTypes.RAW].voxel_size).get_shape()[0]):
                raw.data[z] = self.__augment(
                        raw.data[z],
                        np.random.uniform(low=self.scale_min, high=self.scale_max),
                        np.random.uniform(low=self.shift_min, high=self.shift_max))
        else:
            raw.data = self.__augment(
                    raw.data,
                    np.random.uniform(low=self.scale_min, high=self.scale_max),
                    np.random.uniform(low=self.shift_min, high=self.shift_max))

        # clip values, we might have pushed them out of [0,1]
        raw.data[raw.data>1] = 1
        raw.data[raw.data<0] = 0

    def __augment(self, a, scale, shift):

        return a.mean() + (a-a.mean())*scale + shift
