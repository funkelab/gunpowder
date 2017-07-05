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

        self.upstream_spec = self.get_upstream_provider().get_spec()
        self.spec = copy.deepcopy(self.upstream_spec)

        self.spec.volumes[VolumeTypes.PRED_AFFINITIES] = self.spec.volumes[VolumeTypes.RAW]

        self.worker.start()

    def get_spec(self):
        return self.spec

    def teardown(self):
        self.worker.stop()

    def prepare(self, request):

        self.stored_request = copy.deepcopy(request)

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

        batch.volumes[VolumeTypes.PRED_AFFINITIES]     = out.volumes[VolumeTypes.PRED_AFFINITIES]
        batch.volumes[VolumeTypes.PRED_AFFINITIES].roi = self.stored_request.volumes[VolumeTypes.PRED_AFFINITIES]


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
                'data': batch.volumes[VolumeTypes.RAW].data[np.newaxis,np.newaxis,:],
        })

        loss = self.net.forward()
        output = self.net_io.get_outputs()
        assert len(output['aff_pred'].shape) == 5, "Got affinity prediction with unexpected number of dimensions, should be 1 (direction) + 3 (spatial) + 1 (batch, not used), but is %d"%len(output['aff_pred'].shape)
        output_shape = output['aff_pred'][0][0].shape
        batch.volumes[VolumeTypes.PRED_AFFINITIES] = Volume(data=output['aff_pred'][0],
                                                           roi=Roi((0,)*len(output_shape), output_shape),
                                                           resolution=batch.volumes[VolumeTypes.RAW].resolution)
        return batch
