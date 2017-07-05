import logging
import multiprocessing
import numpy as np
import time

from gunpowder.caffe.net_io_wrapper import NetIoWrapper
from gunpowder.ext import caffe
from gunpowder.nodes.batch_filter import BatchFilter
from gunpowder.producer_pool import ProducerPool, WorkersDied
from gunpowder.volume import VolumeTypes, Volume

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

    def prepare(self, request):

        # remove request parts that we provide
        for volume_type in [VolumeTypes.LOSS_GRADIENT, VolumeTypes.PRED_AFFINITIES]:
            if volume_type in request.volumes:
                del request.volumes[volume_type]

        # for Euclidean training, require a loss scale volume
        if self.solver_parameters.train_state.get_stage(0) == 'euclid':
            request.volumes[VolumeTypes.LOSS_SCALE] = request.volumes[VolumeTypes.GT_AFFINITIES]

    def process(self, batch, request):

        self.batch_in.put((batch,request))

        try:
            out = self.worker.get()
        except WorkersDied:
            raise TrainProcessDied()

        batch.volumes[VolumeTypes.PRED_AFFINITIES] = out.volumes[VolumeTypes.PRED_AFFINITIES]
        if VolumeTypes.LOSS_GRADIENT in request.volumes:
            batch.volumes[VolumeTypes.LOSS_GRADIENT] = out.volumes[VolumeTypes.LOSS_GRADIENT]
        batch.loss = out.loss
        batch.iteration = out.iteration

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

        batch, request = self.batch_in.get()

        data = {
            'data': batch.volumes[VolumeTypes.RAW].data[np.newaxis,np.newaxis,:],
            'aff_label': batch.volumes[VolumeTypes.GT_AFFINITIES].data[np.newaxis,:],
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
        batch.volumes[VolumeTypes.PRED_AFFINITIES] = Volume(
                output['aff_pred'][0],
                batch.volumes[VolumeTypes.GT_AFFINITIES].roi,
                batch.volumes[VolumeTypes.GT_AFFINITIES].resolution
        )
        batch.loss = loss
        batch.iteration = self.solver.iter
        if VolumeTypes.LOSS_GRADIENT in request.volumes:
            diffs = self.net_io.get_output_diffs()
            batch.volumes[VolumeTypes.LOSS_GRADIENT] = Volume(
                    diffs['aff_pred'][0],
                    batch.volumes[VolumeTypes.GT_AFFINITIES].roi,
                    batch.volumes[VolumeTypes.GT_AFFINITIES].resolution
            )

        time_of_iteration = time.time() - start
        logger.info("Train process: iteration=%d loss=%f time=%f"%(self.solver.iter,batch.loss,time_of_iteration))

        return batch

    def __prepare_euclidean(self, batch, data):

        data['scale'] = batch.volumes[VolumeTypes.LOSS_SCALE].data[np.newaxis,:]

    def __prepare_malis(self, batch, data):

        gt_labels = batch.volumes[VolumeTypes.GT_LABELS]
        next_id = gt_labels.data.max() + 1

        gt_pos_pass = gt_labels.data

        if VolumeTypes.GT_IGNORE in batch.volumes:

            gt_neg_pass = np.array(gt_labels.data)
            gt_neg_pass[batch.volumes[VolumeTypes.GT_IGNORE].data==0] = next_id

        else:

            gt_neg_pass = gt_pos_pass

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
