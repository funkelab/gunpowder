from gunpowder import BatchProvider, GraphKey, Graph, ArrayKey, Array, Batch

import copy


class ArraySource(BatchProvider):
    def __init__(self, key: ArrayKey, array: Array):
        self.key = key
        self.array = array

    def setup(self):
        self.provides(self.key, self.array.spec.copy())

    def provide(self, request):
        outputs = Batch()
        if self.array.spec.nonspatial:
            outputs[self.key] = copy.deepcopy(self.array)
        else:
            outputs[self.key] = copy.deepcopy(self.array.crop(request[self.key].roi))
        return outputs


class GraphSource(BatchProvider):
    def __init__(self, key: GraphKey, graph: Graph):
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
