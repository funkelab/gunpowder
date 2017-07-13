import copy
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
from gunpowder.volume import VolumeTypes, Volume

logger = logging.getLogger(__name__)

class PredictProcessDied(Exception):
    pass

class Predict(BatchFilter):
    '''Augments the batch with the predicted affinities.
    '''

    def __init__(self, prototxt, weights, names_net_outputs, use_gpu=None):

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
	self.names_net_outputs = names_net_outputs

    def setup(self):

        self.upstream_spec = self.get_upstream_provider().get_spec()
        self.spec = copy.deepcopy(self.upstream_spec)

        self.spec.volumes[VolumeTypes.PRED_AFFINITIES] = self.spec.volumes[VolumeTypes.RAW]

        self.worker.start()

    def get_spec(self):
        return self.spec

    def teardown(self):
        self.worker.stop()

    def prepare(self, request):

        # remove request parts that node will provide
        for volume_type in [VolumeTypes.PRED_AFFINITIES]:
            if volume_type in request.volumes:
                del request.volumes[volume_type]

    def process(self, batch, request):

        self.batch_in.put(batch)

        try:
            out = self.worker.get()
        except WorkersDied:
            raise PredictProcessDied()

        affs = out.volumes[VolumeTypes.PRED_AFFINITIES]
        affs.roi = request.volumes[VolumeTypes.PRED_AFFINITIES]
        affs.resolution = batch.volumes[VolumeTypes.RAW].resolution

        batch.volumes[VolumeTypes.PRED_AFFINITIES] = affs

    def __predict(self, use_gpu):

        if not self.net_initialized:

            logger.info("Initializing solver...")

            if use_gpu is not None:

                logger.debug("Predict process: using GPU %d"%use_gpu)
                caffe.enumerate_devices(False)
                caffe.set_devices((use_gpu,))
                caffe.set_mode_gpu()
                caffe.select_device(use_gpu, False)

            self.net = caffe.Net(self.prototxt, self.weights, caffe.TEST)
            self.net_io = NetIoWrapper(self.net,  self.names_net_outputs)
            self.net_initialized = True

        start = time.time()

        batch = self.batch_in.get()

        fetch_time = time.time() - start

        self.net_io.set_inputs({
                'data': batch.volumes[VolumeTypes.RAW].data[np.newaxis,np.newaxis,:],
        })
        self.net.forward()
        output = self.net_io.get_outputs()

        predict_time = time.time() - start

        logger.info("Predict process: time=%f (including %f waiting for batch)"%(predict_time, fetch_time))

        assert len(output['aff_pred'].shape) == 5, "Got affinity prediction with unexpected number of dimensions, should be 1 (direction) + 3 (spatial) + 1 (batch, not used), but is %d"%len(output['aff_pred'].shape)
        batch.volumes[VolumeTypes.PRED_AFFINITIES] = Volume(output['aff_pred'][0], Roi(), (1,1,1))

        return batch
