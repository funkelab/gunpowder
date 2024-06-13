import copy
from typing import TYPE_CHECKING

from .batch_provider import BatchProvider

if TYPE_CHECKING:
    from gunpowder import Array, ArrayKey, Batch


class ArraySource(BatchProvider):
    def __init__(self, key: "ArrayKey", array: "Array"):
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
