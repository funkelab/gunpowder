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

    def __init__(self, prototxt, weights, input_names_to_types, output_names_to_types, use_gpu=None):
        '''
        :param prototxt:                network prototxt file
        :param weights:                 network weights file
        :param input_names_to_types:    dict, mapping name of input volume(s) to VolumeType of input volume(s)
                                        (e.g. {'data': VolumeType.RAW}
        :param output_names_to_types:   dict, mapping name of output volume(s) to VolumeType of output volume(s)
                                        (e.g. {'aff_pred': VolumeTypes.PRED_AFFINITIES})
        :param use_gpu:                 int, gpu to use
        '''

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
        self.input_names_to_types  = input_names_to_types
        self.output_names_to_types = output_names_to_types

    def setup(self):

        self.upstream_spec = self.get_upstream_provider().get_spec()
        self.spec = copy.deepcopy(self.upstream_spec)

        for output_name, output_type in self.output_names_to_types.items():
            self.spec.volumes[output_type] = self.spec.volumes[VolumeTypes.RAW]

        self.worker.start()

    def get_spec(self):
        return self.spec

    def teardown(self):
        self.worker.stop()

    def prepare(self, request):
        self.stored_request = copy.deepcopy(request)

        # remove request parts that we provide
        for output_name, output_type in self.output_names_to_types.items():
            if output_type in request.volumes:
                del request.volumes[output_type]

    def process(self, batch, request):

        self.batch_in.put(batch)

        try:
            out = self.worker.get()
        except WorkersDied:
            raise PredictProcessDied()

        for output_name, output_type in self.output_names_to_types.items():
            batch.volumes[output_type]     = out.volumes[output_type]
            batch.volumes[output_type].roi = self.stored_request.volumes[output_type]


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
            self.net_io = NetIoWrapper(self.net,  self.output_names_to_types.keys())
            self.net_initialized = True

        start = time.time()

        batch = self.batch_in.get()

        fetch_time = time.time() - start

        for input_volume_name, input_volume_type in self.input_names_to_types.items():
            self.net_io.set_inputs({input_volume_name: batch.volumes[input_volume_type].data[np.newaxis,np.newaxis,:]})

        self.net.forward()
        output = self.net_io.get_outputs()

        predict_time = time.time() - start

        logger.info("Predict process: time=%f (including %f waiting for batch)" % (predict_time, fetch_time))

        for output_name, output_type in self.output_names_to_types.items():
            assert len(output[output_name].shape) == 5, "Got prediction with unexpected number of dimensions, should be 1 (direction) + 3 (spatial) + 1 (batch, not used), but is %d" % len(
                output[output_name].shape)
            output_shape = output[output_name][0][0].shape
            batch.volumes[output_type] = Volume(data=output[output_name][0],
                                                               roi=Roi((0,) * len(output_shape), output_shape),
                                                               resolution=batch.volumes[VolumeTypes.RAW].resolution)
        return batch
