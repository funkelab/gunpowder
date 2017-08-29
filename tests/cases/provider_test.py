from gunpowder import *
import copy
import unittest
import numpy as np

class TestSource(BatchProvider):

    def setup(self):

        self.provides(
            VolumeTypes.RAW,
            VolumeSpec(
                roi=Roi((0, 0, 0), (100, 100, 100)),
                voxel_size=Coordinate((1, 1, 1)),
                dtype=np.uint8,
                interpolatable=True))

    def provide(self, request):

        data = np.zeros(
            request[VolumeTypes.RAW].roi.get_shape()/self.spec[VolumeTypes.RAW].voxel_size,
            dtype=np.uint8)
        spec = copy.deepcopy(self.spec[VolumeTypes.RAW])
        spec.roi = request[VolumeTypes.RAW].roi

        batch = Batch()
        batch.volumes[VolumeTypes.RAW] = Volume(data, spec)
        return batch


class ProviderTest(unittest.TestCase):

    def setUp(self):

        self.test_source = TestSource()
        self.test_request = BatchRequest()
        self.test_request[VolumeTypes.RAW] = VolumeSpec(
            roi=Roi((20, 20, 20),(10, 10, 10)))
