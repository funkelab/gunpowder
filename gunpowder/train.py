from ext import caffe
from net_input_wrapper import NetInputWrapper
import numpy as np
import time

def init_solver(solver_parameters, device):
    # TODO: are the following two lines needed here?
    caffe.set_mode_gpu()
    caffe.select_device(device, False)

    return caffe.get_solver(solver_parameters)

def train(solver, device, batch_provider):

    caffe.select_device(device, False)

    net_io = NetInputWrapper(solver.net)

    batch_provider.initialize_all()

    # Loop from current iteration to last iteration
    for i in range(solver.iter, solver.max_iter):

        start = time.time()

        batch = batch_provider.request_batch(None)
        data = {
            'data': batch.raw[np.newaxis,:],
            'labels': batch.gt[np.newaxis,:],
            'components': batch.gt_affinities[np.newaxis,:],
        }
        if batch.gt_mask is not None:
            data['scale'] = batch.gt_mask[np.newaxis,:]

        net_io.set_inputs(data)

        loss = solver.step(1)  # Single step
        while gc.collect():
            pass
        time_of_iteration = time.time() - start
