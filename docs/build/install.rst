.. _sec_install:

Installing gunpowder
====================

.. automodule:: gunpowder

``gunpowder`` can be installed in several ways, including using *conda* or direct installation from *github* or by using 
``gunpowder`` *docker* image.


Installing using *conda*
------------------------

Installing ``gunpowder`` using *conda* is simple and straightforward. After create a new *conda* environment (e.g., called gp), 
install *python* (``gunpowder`` works with python3 and python2) and ``gunpowder``

.. code-block:: bash
    
    conda create --name gp python
    source activate gp
    pip install gunpowder

To see if it's successfully installed, type in terminal:

.. code-block:: bash

    conda list | grep "gunpowder"


Using ``gunpowder`` *docker* image
----------------------------------

We created a ``gunpowder`` *docker* image for pulling from *dockerhub*. Follow listed steps to use ``gunpowder`` *docker* container:

1. Install `docker <https://docs.docker.com/install/>`_ 

2. For GPU support, first do `repository configuration <https://nvidia.github.io/nvidia-docker/>`_ and then install `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ 

3. Pull ``gunpowder`` *docker* image from *dockerhub* respository

4. Run ``gunpowder`` image as container

.. code-block:: bash

    docker pull funkey/gunpowder
    nvidia-docker run --rm funkey/gunpowder


Install from *github*
---------------------

``gunpowder`` can also be installed from *github*. After downloading ``gunpowder`` from `github <https://github.com/funkey/gunpowder/tree/release-v1.0>`_, 
run *setup.py* file in ``gunpowder`` folder:

.. code-block:: bash

    pip install .

To see if it's successfully installed, type in terminal:

.. code-block:: bash

    pip list | grep "gunpowder"