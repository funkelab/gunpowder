
class GenericJaxModel():

    def __init__(self):
        pass

    def initialize(self, rng_key, inputs, is_training):
        raise RuntimeError("Unimplemented")

    def forward(self, params, inputs):
        raise RuntimeError("Unimplemented")

    def train_step(self, params, inputs, pmapped=False):
        raise RuntimeError("Unimplemented")
