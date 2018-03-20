import logging
import numpy as np

logger = logging.getLogger(__name__)

def get_net_input_specs(net):
    input_specs = {}
    for layer in net.layers:
        if layer.type == 'MemoryData':
            for i in range(0, layer.layer_param.top_size):
                blob_name = layer.layer_param.get_top(i)
                blob = net.blobs[blob_name]
                input_spec = InputSpec(blob_name, layer, blob, np.shape(blob.data))
                input_specs[input_spec.name] = input_spec
    return input_specs

def get_net_output_specs(net, blob_names):
    output_specs = {}
    for blob_name in blob_names:
        output_spec = OutputSpec(blob_name, net.blobs[blob_name], np.shape(net.blobs[blob_name].data))
        output_specs[output_spec.name] = output_spec
    return output_specs

class InputSpec(object):

    def __init__(self, name, memory_layer, blob, shape, phase=0):
        self.name = name
        self.memory_layer = memory_layer
        self.blob = blob
        self.shape = shape
        self.phase = phase # only added to the network

class OutputSpec(object):

    def __init__(self, name, blob, shape):
        self.name = name
        self.blob = blob
        self.shape = shape

# Wrapper around a networks set_input_arrays to prevent memory leaks of locked
# up arrays
class NetIoWrapper:

    def __init__(self, net, names_net_outputs):
        '''
        :param net:                network prototxt file
        :param names_net_outputs:  list, names of network outputs
        '''
        self.net = net
        self.input_specs = get_net_input_specs(net)
        self.output_specs = get_net_output_specs(net, names_net_outputs)
        self.inputs = {}

        for set_key in self.input_specs.keys():
            shape = self.input_specs[set_key].shape
            # Pre-allocate arrays that will persist with the network
            self.inputs[set_key] = np.zeros(tuple(shape), dtype=np.float32)

    def set_inputs(self, data):
        for set_key in self.input_specs.keys():
            try:
                np.copyto(self.inputs[set_key], np.ascontiguousarray(data[set_key]).astype(np.float32))
                self.net.set_layer_input_arrays(self.input_specs[set_key].memory_layer, self.inputs[set_key], None)
            except:
                logger.error("Could not set input '%s':"%set_key)
                raise

    def get_outputs(self):
        outputs = {}
        for set_key in self.output_specs.keys():
            outputs[set_key] = self.output_specs[set_key].blob.data
        return outputs

    def get_output_diffs(self):
        diffs = {}
        for set_key in self.output_specs.keys():
            diffs[set_key] = self.output_specs[set_key].blob.diff
        return diffs
