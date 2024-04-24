from __future__ import print_function

import pytest

from gunpowder import GraphKey, GraphKeys


def test_register():
    GraphKey("TEST_GRAPH")

    assert GraphKeys.TEST_GRAPH
    with pytest.raises(AttributeError):
        getattr(GraphKeys, "TEST_GRAPH_2")
