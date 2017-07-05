from __future__ import print_function
from gunpowder import *
import unittest

class TestVolumeTypes(unittest.TestCase):

    def test_register(self):

        new_type = VolumeType('TEST_VOLUME1', interpolate=False)
        register_volume_type(new_type)
        new_type = VolumeType('TEST_VOLUME2', interpolate=True)
        register_volume_type(new_type)

        print("pre-registered volume type:", VolumeTypes.RAW, "interpolatable:", VolumeTypes.RAW.interpolate)
        print("new regietered volume type:", VolumeTypes.TEST_VOLUME1, "interpolatable:", VolumeTypes.TEST_VOLUME1.interpolate)
        print("new regietered volume type:", VolumeTypes.TEST_VOLUME2, "interpolatable:", VolumeTypes.TEST_VOLUME2.interpolate)

        self.assertTrue(VolumeTypes.RAW.interpolate)
        self.assertFalse(VolumeTypes.TEST_VOLUME1.interpolate)
        self.assertTrue(VolumeTypes.TEST_VOLUME2.interpolate)
