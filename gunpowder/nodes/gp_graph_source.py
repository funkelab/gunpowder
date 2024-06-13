import copy
from typing import TYPE_CHECKING

from .batch_provider import BatchProvider

if TYPE_CHECKING:
    from gunpowder import Batch, Graph, GraphKey


class GraphSource(BatchProvider):
    def __init__(self, key: "GraphKey", graph: "Graph"):
        self.key = key
        self.graph = graph

    def setup(self):
        self.provides(self.key, self.graph.spec)

    def provide(self, request):
        outputs = Batch()
        outputs[self.key] = copy.deepcopy(
            self.graph.crop(request[self.key].roi).trim(request[self.key].roi)
        )
        return outputs
