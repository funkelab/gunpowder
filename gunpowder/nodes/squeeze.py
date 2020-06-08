import copy
from typing import List
import logging

import numpy as np

from gunpowder.array import ArrayKey
from gunpowder.batch_request import BatchRequest
from gunpowder.batch import Batch
from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)


class Squeeze(BatchFilter):
    """Squeeze a batch at a given axis

    Args:
        arrays (List[ArrayKey]): ArrayKeys to squeeze.
        axis: Position of the single-dimensional axis to remove, defaults to 0.
    """

    def __init__(self, arrays: List[ArrayKey], axis: int = 0):
        self.arrays = arrays
        self.axis = axis

        if self.axis != 0:
            raise NotImplementedError(
                'Squeeze only supported for leading dimension')

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
            outputs[array].data = np.squeeze(batch[array].data, self.axis)
            logger.debug(f'{array} shape: {outputs[array].data.shape}')

        return outputs
