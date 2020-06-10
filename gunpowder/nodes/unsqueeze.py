import copy
from typing import List
import logging

import numpy as np

from gunpowder.array import ArrayKey
from gunpowder.batch import Batch
from gunpowder.batch_request import BatchRequest
from .batch_filter import BatchFilter


logger = logging.getLogger(__name__)


class Unsqueeze(BatchFilter):
    """Unsqueeze a batch at a given axis

    Args:
        arrays (List[ArrayKey]): ArrayKeys to unsqueeze.
        axis: Position where the new axis is placed, defaults to 0.
    """

    def __init__(self, arrays: List[ArrayKey], axis: int = 0):
        self.arrays = arrays
        self.axis = axis

        if self.axis != 0:
            raise NotImplementedError(
                'Unsqueeze only supported for leading dimension')

    def setup(self):
        self.enable_autoskip()
        for array in self.arrays:
            self.updates(array, self.spec[array].copy())

    def prepare(self, request):
        deps = BatchRequest()
        for array in self.arrays:
            deps[array] = request[array].copy()
        return deps

    def process(self, batch, request):
        outputs = Batch()
        for array in self.arrays:
            outputs[array] = copy.deepcopy(batch[array])
            outputs[array].data = np.expand_dims(batch[array].data, self.axis)
        return outputs
