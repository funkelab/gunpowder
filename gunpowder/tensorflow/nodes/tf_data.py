import collections
import copy
import logging
import multiprocessing

from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.array import Array
from gunpowder.batch import Batch

import tensorflow as tf

logger = logging.getLogger(__name__)


class TFData(BatchFilter):
    '''Creates a tf.data.Dataset from upstream batch.

    Without this, data is moved to the GPU memory using feed_dict,
    which is often slower and does not allow for prefetching.
    Similar to the PreCache node, this node only makes sense if:

    1. Incoming batch requests are repeatedly the same.
    2. There is a source of randomness in upstream nodes.

    Args:

        prefetch_size ('auto' or ``int``):

            How many batches to prefetch to GPU.
            For 3d data more than 2 or 4 seems to be counterproductive.
            Recommendation: 'auto' (tensorflow figures it out on its own)

    '''

    def __init__(self, prefetch_size='auto', batch_size=1):

        if prefetch_size == 'auto':
            self.prefetch_size = tf.data.experimental.AUTOTUNE
        else:
            self.prefetch_size = prefetch_size
        self.batch_size = batch_size
        self.current_request = None
        self.running = True

        self.batch = Batch()

    def teardown(self):

        self.running = False

    def provide(self, request):

        if request != self.current_request:

            assert not request.graph_specs, "cannot handle graph/point specs"

            self.current_request = copy.deepcopy(request)

            def generator_fn():
                while self.running:
                    batch = self.get_upstream_provider().request_batch(self.current_request)
                    inputs = {}
                    for array_key, array in batch.arrays.items():
                        inputs[str(array_key)] = array.data
                    yield inputs

            types = {}
            for key, v in request.items():
                types[str(key)] = tf.as_dtype(request[key].dtype)

            self.dataset = tf.data.Dataset.from_generator(
                generator_fn,
                types)
            if self.batch_size != 1:
                self.dataset = self.dataset.batch(self.batch_size)
            self.dataset = self.dataset.prefetch(self.prefetch_size)
            self.iterator = self.dataset.make_one_shot_iterator()
            self.batch.tf_data = self.iterator.get_next()

            logger.info("setting up tf.data.Dataset done")

        # return batch containing only the states of the requested arrays
        # to pass consistency checks
        Array_State = collections.namedtuple("Array_State",
                                             ["shape", "dtype"])
        for key, s in request.array_specs.items():
            spec = self.spec[key].copy()
            spec.dtype = s.dtype

            shape = None
            if not s.nonspatial:
                spec.roi = s.roi
                spec.voxel_size = s.voxel_size
                if spec.voxel_size is not None:
                    shape = spec.roi.get_shape()//spec.voxel_size
            array_state = Array_State(shape=shape,
                                      dtype=spec.dtype)

            self.batch[key] = Array(array_state, spec, is_empty=True)

        return self.batch
