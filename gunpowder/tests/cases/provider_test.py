from gunpowder import *
import numpy as np
import unittest

class TestSource(BatchProvider):

    def get_spec(self):

        spec = ProviderSpec()
        spec.volumes[VolumeTypes.RAW] = Roi((0,0,0), (100,100,100))
        return spec

    def provide(self, request):

        batch = Batch()
        batch.volumes[VolumeTypes.RAW] = Volume(
                np.zeros(
                        request.volumes[VolumeTypes.RAW].get_shape(),
                        dtype=np.uint8
                ),
                request.volumes[VolumeTypes.RAW],
                (1,1,1),
        )
        return batch


class ProviderTest(unittest.TestCase):

    def setUp(self):

        self.test_source = TestSource()
        self.test_request = BatchRequest()
        self.test_request.volumes[VolumeTypes.RAW] = Roi((20,20,20),(10,10,10))
