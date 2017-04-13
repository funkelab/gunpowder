from ext import caffe
from net_input_wrapper import NetInputWrapper
import numpy as np
import time
from multiprocessing import Process

def init_solver(solver_parameters, device):
    # TODO: are the following two lines needed here?
    caffe.set_mode_gpu()
    caffe.select_device(device, False)

    return caffe.get_solver(solver_parameters)

def train(solver, device, batch_provider):

    # since this might launch processes, we have to do that in the main process
    batch_provider.initialize_all()

    # start training in an own process, so that we can gracefully exit if the 
    # process dies
    process = Process(target=__train, args=(solver, device, batch_provider))
    process.start()
    process.join()

def __train(solver, device, batch_provider):

    caffe.select_device(device, False)

    net_io = NetInputWrapper(solver.net)

    # Loop from current iteration to last iteration
    for i in range(solver.iter, solver.max_iter):

        start = time.time()

        batch = batch_provider.request_batch(None)
        data = {
            'data': batch.raw[np.newaxis,np.newaxis,:],
            'label': batch.gt_affinities[np.newaxis,:],
            'components': batch.gt[np.newaxis,np.newaxis,:],
        }

        if 'scale' in net_io.inputs:
            frac_pos = np.clip(batch.gt_affinities.mean(), 0.05, 0.95)
            w_pos = 1.0 / (2.0 * frac_pos)
            w_neg = 1.0 / (2.0 * (1.0 - frac_pos))
            error_scale = scale_errors(batch.gt_affinities, w_neg, w_pos)
            data['scale'] = error_scale[np.newaxis,:]

        net_io.set_inputs(data)

        loss = solver.step(1)  # Single step
        while gc.collect():
            pass
        time_of_iteration = time.time() - start

def scale_errors(data, factor_low, factor_high):
    scaled_data = np.add((data >= 0.5) * factor_high, (data < 0.5) * factor_low)
    return scaled_data
