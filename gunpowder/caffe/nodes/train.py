import logging
import numpy as np

from gunpowder.caffe.net_io_wrapper import NetIoWrapper
from gunpowder.ext import caffe
from gunpowder.nodes.generic_train import GenericTrain
from gunpowder.volume import Volume, VolumeType

logger = logging.getLogger(__name__)

class Train(GenericTrain):
    '''Caffe implementation of :class:`gunpowder.nodes.Train`.

    Args:

        solver_parameters (:class:``SolverParameters``): Parameters of the
            solver to use for training, contains the network description as
            well.

        inputs (dict): Dictionary from names of input layers in the network to
            :class:``VolumeType`` or batch attribute name as string.

        outputs (dict): Dictionary from the names of output layers in the
            network to :class:``VolumeType``. New volumes will be generated by
            this node for each entry (if requested downstream).

        gradients (dict): Dictionary from the names of output layers in the
            network to :class:``VolumeType``. New volumes containing the
            gradient of an output with respect to the loss will be generated by
            this node for each entry (if requested downstream).

        volume_specs (dict, optional): An optional dictionary of
            :class:`VolumeType` to :class:`VolumeSpec` to set the volume specs
            generated volumes (``outputs`` and ``gradients``). This is useful
            to set the ``voxel_size``, for example, if they differ from the
            voxel size of the input volumes. Only fields that are not ``None``
            in the given :class:`VolumeSpec` will be used.

        use_gpu (int): Which GPU to use. Set to ``None`` for CPU mode.
    '''

    def __init__(
            self,
            solver_parameters,
            inputs,
            outputs,
            gradients,
            volume_specs=None,
            use_gpu=None):

        super(Train, self).__init__(
            inputs,
            outputs,
            gradients,
            volume_specs,
            spawn_subprocess=True)
        self.solver_parameters = solver_parameters
        self.use_gpu = use_gpu
        self.solver = None
        self.net_io = None

    def start(self):

        logger.info("Initializing solver...")

        if self.use_gpu is not None:

            logger.debug("Train process: using GPU %d", self.use_gpu)
            caffe.enumerate_devices(False)
            caffe.set_devices((self.use_gpu,))
            caffe.set_mode_gpu()
            caffe.select_device(self.use_gpu, False)

        self.solver = caffe.get_solver(self.solver_parameters)
        if self.solver_parameters.resume_from is not None:
            logger.debug(
                "Train process: restoring solver state from %s",
                self.solver_parameters.resume_from)
            self.solver.restore(self.solver_parameters.resume_from)

        names_net_outputs = self.outputs.keys() + self.gradients.keys()
        self.net_io = NetIoWrapper(self.solver.net, names_net_outputs)

    def train_step(self, batch, request):

        data = {}
        for input_name, network_input in self.inputs.items():
            if isinstance(network_input, VolumeType):
                data[input_name] = batch.volumes[network_input].data
            elif isinstance(network_input, np.ndarray):
                data[input_name] = network_input
            elif isinstance(network_input, str):
                data[input_name] = getattr(batch, network_input)
            else:
                raise Exception(
                    "Unknown network input type {}, can't be given to "
                    "network".format(network_input))
        self.net_io.set_inputs(data)

        loss = self.solver.step(1)
        # self.__consistency_check()

        requested_outputs = {
            name: volume_type
            for name, volume_type in self.outputs.items()
            if volume_type in request.volume_specs }

        if requested_outputs:

            output = self.net_io.get_outputs()

            for output_name, volume_type in requested_outputs.items():

                spec = self.spec[volume_type].copy()
                spec.roi = request[volume_type].roi
                batch.volumes[volume_type] = Volume(
                    output[output_name][0], # strip #batch dimension
                    spec)

        requested_gradients = {
            name: volume_type
            for name, volume_type in self.gradients.items()
            if volume_type in request.volume_specs }

        if requested_gradients:

            diffs = self.net_io.get_output_diffs()

            for output_name, volume_type in requested_gradients.items():

                spec = self.spec[volume_type].copy()
                spec.roi = request[volume_type].roi
                batch.volumes[volume_type] = Volume(
                    diffs[output_name][0], # strip #batch dimension
                    spec)

        batch.loss = loss
        batch.iteration = self.solver.iter

    def __consistency_check(self):

        diffs = self.net_io.get_output_diffs()
        for k in diffs:
            assert not np.isnan(diffs[k]).any(), "Detected NaN in output diff " + k
