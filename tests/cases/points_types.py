from __future__ import print_function
from gunpowder import *
import unittest

class TestPointsTypes(unittest.TestCase):

    def test_register(self):

        register_points_type('TEST_POINTS1')

        print("pre-registered points type:", PointsTypes.PRESYN)
        print("new registered points type:", PointsTypes.TEST_POINTS1)

        self.assertTrue(PointsTypes.PRESYN)
        self.assertTrue(PointsTypes.TEST_POINTS1)
        self.assertRaises(AttributeError, getattr, PointsTypes, "TEST_POINTS2")
