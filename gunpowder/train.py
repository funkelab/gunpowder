import multiprocessing
import atexit
import time
import numpy as np
from net_input_wrapper import NetInputWrapper
from ext import caffe
from batch_filter import BatchFilter

class TrainProcessDied(Exception):
    pass

class Train(BatchFilter):
    '''If a ground-truth mask is set and is not 1 everywhere, creates two 
    versions of the ground-truth labels, one for the positive and one for the 
    negative Malis pass.

    Use this filter after your ground-truth and mask are finalized.
    '''

    def __init__(self, solver_parameters, use_gpu=None, dry_run=False):

        # TODO: are the following two lines needed here?
        if use_gpu is not None:
            caffe.set_mode_gpu()
            caffe.select_device(use_gpu, False)

        self.solver = caffe.get_solver(solver_parameters)
        self.batch_in = multiprocessing.Queue(maxsize=1)
        self.batch_out = multiprocessing.Queue(maxsize=1)
        self.stopped = None

        # start training in an own process, so that we can gracefully exit if 
        # the process dies
        self.train_process = multiprocessing.Process(target=self.__train, args=(self.solver, use_gpu))

        self.dry_run = dry_run

    def initialize(self):

        if self.stopped is None:
            self.stopped = multiprocessing.Event()
            self.stopped.clear()
            self.train_process.start()
            atexit.register(self.__del__)

    def __del__(self):

        print("Train: being killed")
        self.train_process.terminate()

    def process(self, batch):

        print("Train: sending batch to training process...")
        self.batch_in.put(batch)
        print("Train: sent, waiting for result...")
        out = None
        while out is None:
            # print("Train: current train batch is " + str(out))
            try:
                out = self.batch_out.get(timeout=1)
            except:
                # print("Train: output queue is still empty")
                if not self.train_process.is_alive():
                    print("Train: training process is not alive anymore")
                    raise TrainProcessDied()
        print("Train: got training result")

        batch.prediction = out.prediction
        batch.gradient = out.gradient
        batch.loss = out.loss

    def __train(self, solver, use_gpu):

        if use_gpu is not None:
            caffe.select_device(use_gpu, False)

        net_io = NetInputWrapper(solver.net)

        while not self.stopped.is_set():

            start = time.time()

            print("Train process: waiting for batch...")
            batch = self.batch_in.get()
            data = {
                'data': batch.raw[np.newaxis,np.newaxis,:],
                'label': batch.gt_affinities[np.newaxis,:],
                'components': batch.gt[np.newaxis,np.newaxis,:],
            }

            if 'scale' in net_io.inputs:
                self.__prepare_euclidean(batch, data)
            else:
                self.__prepare_malis(batch, data)

            net_io.set_inputs(data)

            if self.dry_run:
                batch.prediction = np.zeros((3,) + batch.spec.shape, dtype=np.float32)
                batch.gradient = np.zeros((3,) + batch.spec.shape, dtype=np.float32)
                batch.loss = 0
            else:
                loss = solver.step(1)
                # TODO: add prediction, gradient, loss

            time_of_iteration = time.time() - start

            self.batch_out.put(batch)
            print("Train process: finished batch")

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

        if batch.gt_mask is None:
            return

        if batch.gt_mask.mean() == 1:
            return

        next_id = batch.gt.max() + 1

        gt_pos_pass = batch.gt
        gt_neg_pass = np.array(batch.gt)
        gt_neg_pass[batch.gt_mask==0] = next_id

        data['components'] = np.array([gt_neg_pass, gt_pos_pass])

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
