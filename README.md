![gunpowder](docs/_static/gunpowder.svg)

[![Build Status](https://travis-ci.org/funkey/gunpowder.svg?branch=master)](https://travis-ci.org/funkey/gunpowder)

A library to facilitate machine learning on large, multi-dimensional images.

`gunpowder` allows you to assemble a pipeline from
[data loading](http://funkey.science/gunpowder/api.html#source-nodes)
over
[pre-processing](http://funkey.science/gunpowder/api.html#image-processing-nodes),
[random batch sampling](http://funkey.science/gunpowder/api.html#randomlocation),
[data augmentation](http://funkey.science/gunpowder/api.html#augmentation-nodes),
[pre-caching](http://funkey.science/gunpowder/api.html#precache),
[training/prediction](http://funkey.science/gunpowder/api.html#training-and-prediction-nodes), to
[storage of results](http://funkey.science/gunpowder/api.html#output-nodes)
on arbitrarily large volumes of
multi-dimensional images. `gunpowder` is not tied to a particular learning
framework, and thus complements libraries like
[`torch`](https://pytorch.org/),
[`tensorflow`](https://www.tensorflow.org/).

The full documentation can be found [here](https://funkey.github.io/gunpowder).

`gunpowder` was originally written by Jan Funke and is inspired by
[`PyGreentea`](https://github.com/TuragaLab/PyGreentea) by William Grisaitis,
Fabian Tschopp, and Srini Turaga.
