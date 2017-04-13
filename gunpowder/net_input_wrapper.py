import numpy as np

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

class InputSpec(object):
    def __init__(self, name, memory_layer, blob, shape, phase=0):
        self.name = name
        self.memory_layer = memory_layer
        self.blob = blob
        self.shape = shape
        self.phase = phase # only added to the network  

# Wrapper around a networks set_input_arrays to prevent memory leaks of locked 
# up arrays
class NetInputWrapper:

    def __init__(self, net):
        input_specs = get_net_input_specs(net)
        self.net = net
        self.input_specs = input_specs
        self.inputs = {}

        for set_key in self.input_specs.keys():
            shape = self.input_specs[set_key].shape
            # Pre-allocate arrays that will persist with the network
            self.inputs[set_key] = np.zeros(tuple(shape), dtype=np.float32)

    def set_inputs(self, data):
        for set_key in self.input_specs.keys():
            np.copyto(self.inputs[set_key], np.ascontiguousarray(data[set_key]).astype(np.float32))
            self.net.set_layer_input_arrays(self.input_specs[set_key].memory_layer, self.inputs[set_key], None)
