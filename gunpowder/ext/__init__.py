class NoSuchModule(object):

    def __init__(self, name):
        self.__name = name

    def __getattr__(self, item):
        raise ImportError('Module {0} is not installed'.format(self.__name))

try:
    import dvision
except ImportError:
    dvision = NoSuchModule('dvision')

try:
    import h5py
except ImportError:
    h5py = NoSuchModule('h5py')

try:
    import caffe
except ImportError:
    caffe = NoSuchModule('caffe')

try:
    import tensorflow
except ImportError:
    tensorflow = NoSuchModule('tensorflow')

try:
    import malis
except ImportError:
    malis = NoSuchModule('malis')

try:
    import augment
except ImportError:
    augment = NoSuchModule('augment')
