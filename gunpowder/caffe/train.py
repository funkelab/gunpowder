import multiprocessing
import time
import numpy as np
import os
from net_io_wrapper import NetIoWrapper
import caffe
from ..batch_filter import BatchFilter

import logging
logger = logging.getLogger(__name__)

class TrainProcessDied(Exception):
    pass

class Train(BatchFilter):
    '''Performs one training iteration for each batch that passes through. 
    Augments the batch with the predicted affinities.
    '''

    def __init__(self, solver_parameters, use_gpu=None, dry_run=False):

        self.batch_in = multiprocessing.Queue(maxsize=1)
        self.batch_out = multiprocessing.Queue(maxsize=1)

        # start training in an own process, so that we can gracefully exit if 
        # the process dies
        self.train_process = multiprocessing.Process(target=self.__train, args=(use_gpu,))

        self.dry_run = dry_run
        self.solver_parameters = solver_parameters

    def setup(self):
        self.__start_worker()

    def teardown(self):
        self.__stop_worker()

    def process(self, batch):

        self.batch_in.put(batch)

        out = None
        while out is None:

            try:

                out = self.batch_out.get(timeout=1)

            except:

                if not self.train_process.is_alive():

                    logger.error("Train: training process is not alive anymore")
                    self.__stop_worker()
                    raise TrainProcessDied()

        batch.prediction = out.prediction
        batch.gradient = out.gradient
        batch.loss = out.loss

    def __del__(self):
        self.__stop_worker()

    def __stop_worker(self):

        self.train_process.start()

    def __stop_worker(self):

        logger.info("terminating train process...")
        self.train_process.terminate()
        self.train_process.join()

    def __train(self, use_gpu):

        if use_gpu is not None:

            logger.debug("Train process: using GPU %d"%use_gpu)
            caffe.enumerate_devices(False)
            caffe.set_devices((use_gpu,))
            caffe.set_mode_gpu()
            caffe.select_device(use_gpu, False)

        solver = caffe.get_solver(self.solver_parameters)
        if self.solver_parameters.resume_from is not None:
            logger.debug("Train process: restoring solver state from " + self.solver_parameters.resume_from)
            solver.restore(self.solver_parameters.resume_from)

        net_io = NetIoWrapper(solver.net)

        while not self.__parent_died():

            start = time.time()

            batch = self.batch_in.get()
            data = {
                'data': batch.raw[np.newaxis,np.newaxis,:],
                'aff_label': batch.gt_affinities[np.newaxis,:],
            }

            if self.solver_parameters.train_state.get_stage(0) == 'euclid':
                logger.debug("Train process: preparing input data for Euclidean training")
                self.__prepare_euclidean(batch, data)
            else:
                logger.debug("Train process: preparing input data for Malis training")
                self.__prepare_malis(batch, data)

            net_io.set_inputs(data)

            if self.dry_run:
                logger.warning("Train process: DRY RUN, LOSS WILL BE 0")
                batch.prediction = np.zeros((3,) + batch.spec.shape, dtype=np.float32)
                batch.gradient = np.zeros((3,) + batch.spec.shape, dtype=np.float32)
                batch.loss = 0
            else:
                loss = solver.step(1)
                output = net_io.get_outputs()
                batch.prediction = output['aff_pred']
                batch.loss = loss
                # TODO: add gradient

            time_of_iteration = time.time() - start

            while not self.__parent_died():

                try:
                    self.batch_out.put(batch, timeout=1)
                    logger.info("Train process: iteration=%d loss=%f time=%f"%(solver.iter,batch.loss,time_of_iteration))
                    break
                except:
                    pass

    def __prepare_euclidean(self, batch, data):

        frac_pos = np.clip(batch.gt_affinities.mean(), 0.05, 0.95)
        w_pos = 1.0 / (2.0 * frac_pos)
        w_neg = 1.0 / (2.0 * (1.0 - frac_pos))
        error_scale = self.__scale_errors(batch.gt_affinities, w_neg, w_pos)
        data['scale'] = error_scale[np.newaxis,:]

    def __scale_errors(self, data, factor_low, factor_high):
        scaled_data = np.add((data >= 0.5) * factor_high, (data < 0.5) * factor_low)
        return scaled_data

    def __prepare_malis(self, batch, data):

        next_id = batch.gt.max() + 1

        gt_pos_pass = batch.gt
        gt_neg_pass = np.array(batch.gt)
        gt_neg_pass[batch.gt_mask==0] = next_id

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

    def __parent_died(self):
        return os.getppid() == 1
