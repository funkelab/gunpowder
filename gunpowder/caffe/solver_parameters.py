from gunpowder.ext import caffe, NoSuchModule

try:
    class SolverParameters(caffe.SolverParameter):
        pass
except:
    class SolverParameters(NoSuchModule):
        def __init__(self):
            super(SolverParameters, self).__init__('caffe')
