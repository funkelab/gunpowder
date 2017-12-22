from gunpowder import *
import copy
import unittest
import numpy as np

class TestSource(BatchProvider):

    def setup(self):

        self.provides(
            ArrayKeys.RAW,
            ArraySpec(
                roi=Roi((0, 0, 0), (100, 100, 100)),
                voxel_size=Coordinate((1, 1, 1)),
                dtype=np.uint8,
                interpolatable=True))

    def provide(self, request):

        data = np.zeros(
            request[ArrayKeys.RAW].roi.get_shape()/self.spec[ArrayKeys.RAW].voxel_size,
            dtype=np.uint8)
        spec = copy.deepcopy(self.spec[ArrayKeys.RAW])
        spec.roi = request[ArrayKeys.RAW].roi

        batch = Batch()
        batch.arrays[ArrayKeys.RAW] = Array(data, spec)
        return batch


class ProviderTest(unittest.TestCase):

    def setUp(self):

        self.test_source = TestSource()
        self.test_request = BatchRequest()
        self.test_request[ArrayKeys.RAW] = ArraySpec(
            roi=Roi((20, 20, 20),(10, 10, 10)))
