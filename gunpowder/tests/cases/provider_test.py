from gunpowder import *
import numpy as np
import unittest

class TestSource(BatchProvider):

    def get_spec(self):

        spec = ProviderSpec()
        spec.volumes[VolumeType.RAW] = Roi((0,0,0), (100,100,100))
        return spec

    def request_batch(self, request):

        batch = Batch()
        batch.volumes[VolumeType.RAW] = Volume(
                np.zeros(
                        request.volumes[VolumeType.RAW].get_shape(),
                        dtype=np.uint8
                ),
                request.volumes[VolumeType.RAW],
                (1,1,1),
                True
        )
        return batch


class ProviderTest(unittest.TestCase):

    def setUp(self):

        self.test_source = TestSource()
        self.test_request = BatchRequest()
        self.test_request.volumes[VolumeType.RAW] = Roi((20,20,20),(10,10,10))
