from __future__ import print_function
from gunpowder import GraphKey, GraphKeys
import unittest


class TestGraphKeys(unittest.TestCase):
    def test_register(self):
        GraphKey("TEST_GRAPH")

        self.assertTrue(GraphKeys.TEST_GRAPH)
        self.assertRaises(AttributeError, getattr, GraphKeys, "TEST_GRAPH_2")
