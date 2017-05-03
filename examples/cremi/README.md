Preparation
===========

Download the training samples from the [CREMI Challenge](https://cremi.org/data).

You will also need a `pycaffe` version that has the layers in the `net.prototxt`
implemented. The easiest way to get one is to use the `funkey/gunpowder` docker
image, which already contains `pycaffe` and `gunpowder`.

Run
===

To use the docker image, run the example via

```
./train.sh
```

This will download the docker image, instantiate it via `nvidia-docker`, and
run `train.py`.
