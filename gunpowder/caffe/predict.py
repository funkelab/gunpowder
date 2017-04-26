import multiprocessing
import time
import numpy as np
from net_io_wrapper import NetIoWrapper
import caffe
from ..batch_filter import BatchFilter
from ..producer_pool import ProducerPool, WorkersDied

import logging
logger = logging.getLogger(__name__)

class PredictProcessDied(Exception):
    pass

class Predict(BatchFilter):
    '''Augments the batch with the predicted affinities.
    '''

    def __init__(self, prototxt, weights, use_gpu=None):

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

    def process(self, batch):

        self.batch_in.put(batch)

        try:
            out = self.worker.get()
        except WorkersDied:
            raise PredictProcessDied()

        batch.prediction = out.prediction

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
                'data': batch.raw[np.newaxis,np.newaxis,:],
        })

        loss = self.net.forward()
        output = self.net_io.get_outputs()
        batch.prediction = output['aff_pred']

        return batch
