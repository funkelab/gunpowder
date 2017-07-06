from __future__ import print_function
from gunpowder import *
import unittest

class TestPointsTypes(unittest.TestCase):

    def test_register(self):

        new_type = PointsType('TEST_POINTS1')
        register_points_type(new_type)

        print("pre-registered points type:", PointsTypes.PRESYN)
        print("new registered points type:", PointsTypes.TEST_POINTS1)

        self.assertTrue(PointsTypes.PRESYN)
        self.assertTrue(PointsTypes.TEST_POINTS1)
        self.assertRaises(AttributeError, getattr, PointsTypes, "TEST_POINTS2")
