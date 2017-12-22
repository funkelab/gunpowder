from __future__ import print_function
from gunpowder import *
import unittest

class TestPointsKeys(unittest.TestCase):

    def test_register(self):

        register_points_type('TEST_POINTS1')

        print("pre-registered points type:", PointsKeys.PRESYN)
        print("new registered points type:", PointsKeys.TEST_POINTS1)

        self.assertTrue(PointsKeys.PRESYN)
        self.assertTrue(PointsKeys.TEST_POINTS1)
        self.assertRaises(AttributeError, getattr, PointsKeys, "TEST_POINTS2")
