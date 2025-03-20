import gunpowder as gp
import numpy as np
from gunpowder.nodes import GPGraphSource


def test_gpgraph_source():
    points = gp.GraphKey("points")
    nodes = [
        gp.graph.Node(0, np.array([0, 0, 0, 0])),
        gp.graph.Node(1, np.array([5, 10, 10, 10])),
        gp.graph.Node(2, np.array([10, 50, 50, 50])),
        gp.graph.Node(3, np.array([15, 90, 90, 90])),
    ]
    points_source = GPGraphSource(points, gp.Graph(nodes=nodes, edges=[], spec=gp.GraphSpec()))
    
    request = gp.BatchRequest()
    request_shape = gp.Coordinate((10, 40, 40, 40))
    request_roi = gp.Roi(offset=(5, 30, 30, 30), shape=request_shape)
    points_request = gp.GraphSpec(request_roi,)
    request[points] = points_request

    with gp.build(points_source):
        batch = points_source.request_batch(request)
        points_data = batch[points]
        assert len(list(points_data.nodes)) == 1