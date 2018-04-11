from gunpowder.ext import caffe, NoSuchModule

try:
    class SolverParameters(caffe.SolverParameter):
        '''Wrapper of the caffe's ``SolverParameter`` class.

        Attributes:

            train_net (``string``):

                The network to train.

            base_lr (``float``):

                The initial learning rate.

            snapshot (``int``):

                Weight snapshot interval in iterations.

            snapshot_prefix (``string``)

                Prefix of snapshot files.

            type (``string``):

                Optimizer type, e.g., ``Adam``

                For ``Adam``, the following parameters can be set as attributes
                of this class: ``momentum`` (``float``), ``momentum2``
                (``float``), ``delta`` (``float``), ``weight_decay``
                (``float``), ``lr_policy`` (``string``), ``gamma`` (``float``),
                ``power`` (``float``).

            resume_from (``string`` or ``None``):

                Weight snapshot file to resume training from.

            train_state (training stages):

                Used to set the current trainig stage (if used during
                construction of the network).
        '''
        pass
except:
    class SolverParameters(NoSuchModule):
        '''Wrapper of the caffe's ``SolverParameter`` class.

        Attributes:

            train_net (``string``):

                The network to train.

            base_lr (``float``):

                The initial learning rate.

            snapshot (``int``):

                Weight snapshot interval in iterations.

            snapshot_prefix (``string``)

                Prefix of snapshot files.

            type (``string``):

                Optimizer type, e.g., ``Adam``

                For ``Adam``, the following parameters can be set as attributes
                of this class: ``momentum`` (``float``), ``momentum2``
                (``float``), ``delta`` (``float``), ``weight_decay``
                (``float``), ``lr_policy`` (``string``), ``gamma`` (``float``),
                ``power`` (``float``).

            resume_from (``string`` or ``None``):

                Weight snapshot file to resume training from.

            train_state (training stages):

                Used to set the current trainig stage (if used during
                construction of the network).
        '''
        def __init__(self):
            super(SolverParameters, self).__init__('caffe')
