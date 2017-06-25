import logging
import multiprocessing
import numpy as np
import os
import time

from gunpowder.caffe.net_io_wrapper import NetIoWrapper
from gunpowder.ext import caffe
from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.producer_pool import ProducerPool, WorkersDied
from gunpowder.roi import Roi
from gunpowder.volume import VolumeType, Volume

logger = logging.getLogger(__name__)

class PredictProcessDied(Exception):
    pass

class Predict(BatchFilter):
    '''Augments the batch with the predicted affinities.
    '''

    def __init__(self, prototxt, weights, use_gpu=None):

        for f in [prototxt, weights]:
            if not os.path.isfile(f):
                raise RuntimeError("%s does not exist"%f)

        # start prediction as a producer pool, so that we can gracefully exit if
        # anything goes wrong
        self.worker = ProducerPool([lambda gpu=use_gpu: self.__predict(gpu)], queue_size=1)
        self.batch_in = multiprocessing.Queue(maxsize=1)

        self.prototxt = prototxt
        self.weights = weights
        self.net_initialized = False

    def setup(self):
        self.worker.start()

    def teardown(self):
        self.worker.stop()

    def process(self, batch, request):

        self.batch_in.put(batch)

        try:
            out = self.worker.get()
        except WorkersDied:
            raise PredictProcessDied()

        batch.volumes[VolumeType.PRED_AFFINITIES] = out.volumes[VolumeType.PRED_AFFINITIES]

    def __predict(self, use_gpu):

        start = time.time()

        if not self.net_initialized:

            logger.info("Initializing solver...")

            if use_gpu is not None:

                logger.debug("Predict process: using GPU %d"%use_gpu)
                caffe.enumerate_devices(False)
                caffe.set_devices((use_gpu,))
                caffe.set_mode_gpu()
                caffe.select_device(use_gpu, False)

            self.net = caffe.Net(self.prototxt, self.weights, caffe.TEST)
            self.net_io = NetIoWrapper(self.net)
            self.net_initialized = True

        batch = self.batch_in.get()

        self.net_io.set_inputs({
                'data': batch.volumes[VolumeType.RAW].data[np.newaxis,np.newaxis,:],
        })

        loss = self.net.forward()
        output = self.net_io.get_outputs()
        assert len(output['aff_pred'].shape) == 5, "Got affinity prediction with unexpected number of dimensions, should be 1 (direction) + 3 (spatial) + 1 (batch, not used), but is %d"%len(output['aff_pred'].shape)
        output_shape = output['aff_pred'][0][0].shape
        batch.volumes[VolumeType.PRED_AFFINITIES] = Volume(data=output['aff_pred'][0],
                                                           roi=Roi((0,)*len(output_shape), output_shape),
                                                           resolution=batch.volumes[VolumeType.RAW].resolution,
                                                           interpolate=True)
        return batch
