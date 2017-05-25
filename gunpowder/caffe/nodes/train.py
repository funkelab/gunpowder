import logging
import multiprocessing
import numpy as np
import time

from gunpowder.caffe.net_io_wrapper import NetIoWrapper
from gunpowder.ext import caffe
from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.producer_pool import ProducerPool, WorkersDied

logger = logging.getLogger(__name__)

class TrainProcessDied(Exception):
    pass

class Train(BatchFilter):
    '''Performs one training iteration for each batch that passes through. 
    Adds the predicted affinities to the batch.
    '''

    def __init__(self, solver_parameters, use_gpu=None):

        # start training as a producer pool, so that we can gracefully exit if 
        # anything goes wrong
        self.worker = ProducerPool([lambda gpu=use_gpu: self.__train(gpu)], queue_size=1)
        self.batch_in = multiprocessing.Queue(maxsize=1)

        self.solver_parameters = solver_parameters
        self.solver_initialized = False

    def setup(self):
        self.worker.start()

    def teardown(self):
        self.worker.stop()

    def process(self, batch):

        self.batch_in.put(batch)

        try:
            out = self.worker.get()
        except WorkersDied:
            raise TrainProcessDied()

        batch.volumes[VolumeType.PRED_AFFINITIES] = out.volumes[VolumeType.PRED_AFFINITIES]
        batch.volumes[VolumeType.LOSS_GRADIENT] = out.volumes[VolumeType.LOSS_GRADIENT]
        batch.loss = out.loss

    def __train(self, use_gpu):

        start = time.time()

        if not self.solver_initialized:

            logger.info("Initializing solver...")

            if use_gpu is not None:

                logger.debug("Train process: using GPU %d"%use_gpu)
                caffe.enumerate_devices(False)
                caffe.set_devices((use_gpu,))
                caffe.set_mode_gpu()
                caffe.select_device(use_gpu, False)

            self.solver = caffe.get_solver(self.solver_parameters)
            if self.solver_parameters.resume_from is not None:
                logger.debug("Train process: restoring solver state from " + self.solver_parameters.resume_from)
                self.solver.restore(self.solver_parameters.resume_from)

            self.net_io = NetIoWrapper(self.solver.net)

            self.solver_initialized = True

        batch = self.batch_in.get()

        data = {
            'data': batch.volumes[VolumeType.RAW].data[np.newaxis,np.newaxis,:],
            'aff_label': batch.volumes[VolumeType.GT_AFFINITIES].data[np.newaxis,:],
        }

        if self.solver_parameters.train_state.get_stage(0) == 'euclid':
            logger.debug("Train process: preparing input data for Euclidean training")
            self.__prepare_euclidean(batch, data)
        else:
            logger.debug("Train process: preparing input data for Malis training")
            self.__prepare_malis(batch, data)

        self.net_io.set_inputs(data)

        loss = self.solver.step(1)
        # self.__consistency_check()
        output = self.net_io.get_outputs()
        batch.volumes[VolumeType.PRED_AFFINITIES] = Volume(output['aff_pred'], interpolate=True)
        batch.loss = loss
        # TODO: add gradient

        time_of_iteration = time.time() - start
        logger.info("Train process: iteration=%d loss=%f time=%f"%(self.solver.iter,batch.loss,time_of_iteration))

        return batch

    def __prepare_euclidean(self, batch, data):

        gt_affinities = batch.volumes[VolumeType.GT_AFFINITIES]
        frac_pos = np.clip(gt_affinities.data, 0.05, 0.95)
        w_pos = 1.0 / (2.0 * frac_pos)
        w_neg = 1.0 / (2.0 * (1.0 - frac_pos))
        error_scale = self.__scale_errors(gt_affinities.data, w_neg, w_pos)
        data['scale'] = error_scale[np.newaxis,:]

    def __scale_errors(self, data, factor_low, factor_high):
        scaled_data = np.add((data >= 0.5) * factor_high, (data < 0.5) * factor_low)
        return scaled_data

    def __prepare_malis(self, batch, data):

        gt_labels = batch.volumes[VolumeType.GT_LABELS]
        next_id = gt_labels.data.max() + 1

        gt_pos_pass = gt_labels.data
        gt_neg_pass = np.array(gt_labels.data)
        gt_neg_pass[batch.volumes[VolumeType.GT_MASK].data==0] = next_id

        data['comp_label'] = np.array([[gt_neg_pass, gt_pos_pass]])
        data['nhood'] = batch.affinity_neighborhood[np.newaxis,np.newaxis,:]

        # Why don't we update gt_affinities in the same way?
        # -> not needed
        #
        # GT affinities are all 0 in the masked area (because masked area is 
        # assumed to be set to background in batch.gt).
        #
        # In the negative pass:
        #
        #   We set all affinities inside GT regions to 1. Affinities in masked 
        #   area as predicted. Belongs to one forground region (introduced 
        #   above). But we only count loss on edges connecting different labels 
        #   -> loss in masked-out area only from outside regions.
        #
        # In the positive pass:
        #
        #   We set all affinities outside GT regions to 0 -> no loss in masked 
        #   out area.

    def __consistency_check(self):

        diffs = self.net_io.get_outputs()
        for k in diffs:
            assert not np.isnan(diffs[k]).any(), "Detected NaN in output diff " + k
