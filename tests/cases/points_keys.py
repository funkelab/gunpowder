from __future__ import print_function
from gunpowder import *
import unittest

class TestPointsKeys(unittest.TestCase):

    def test_register(self):

        PointsKey('TEST_POINTS1')

        self.assertTrue(PointsKeys.TEST_POINTS1)
        self.assertRaises(AttributeError, getattr, PointsKeys, "TEST_POINTS2")
