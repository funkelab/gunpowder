import logging
import multiprocessing
import ctypes
from gunpowder.ext import tensorflow as tf
from gunpowder.freezable import Freezable

logger = logging.getLogger(__name__)

class LocalServer(Freezable):
    '''Wrapper around ``tf.train.Server`` to create a local server on-demand.

    This class is necessary because tensorflow's GPU support should not be
    initialized before forking processes (the CUDA driver needs to be
    initialized in each process separately, not in the main process and then
    forked). Creating a ``tf.train.Server`` initializes GPU support, however.
    With this wrapper, server creating can be delayed until a GPU process
    creates a ``tf.Session``::

        session = tf.Session(target=LocalServer.get_target())
    '''

    __target = multiprocessing.Array(ctypes.c_char, b' '*256)
    __server = None

    @staticmethod
    def get_target():
        '''Get the target string of this tensorflow server to connect a
        ``tf.Session()``. This will start the server, if it is not running
        already.
        '''

        with LocalServer.__target.get_lock():

            target = LocalServer.__target.value

            if target == b' '*256:
                logger.info("Creating local tensorflow server")
                LocalServer.__server = tf.train.Server.create_local_server()
                target = LocalServer.__server.target
                logger.info("Server running at %s", target)
            else:
                logger.info("Server already running at %s", target)

            LocalServer.__target.value = target

        return target
