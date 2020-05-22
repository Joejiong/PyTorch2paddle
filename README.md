# PyTorch2Paddle 

[![Build Status](https://travis-ci.org/gzuidhof/nn-transfer.svg?branch=master)](https://travis-ci.org/gzuidhof/nn-transfer)

**NOTE: This repository does not seem to yield the correct output anymore with the latest versions of paddle and keras and PyTorch. Take care to verify the results or use an alternative method for conversion.**

This repository contains utilities for **converting PyTorch models to Keras|paddle and the other way around**. More specifically, it allows you to copy the weights from a PyTorch model to an identical model in Keras|paddle and vice-versa.


## Installation
Clone this repository, and simply run

```
pip install .
```

You need to have PyTorch and torchvision installed beforehand, see the [PyTorch website](https://www.pytorch.org) for how to easily install that.

## Tests

To run the unit and integration tests:

```
python setup.py test
# OR, if you have nose2 installed,
nose2
```

There is also Travis CI which will automatically build every commit, see the button at the top of the readme. You can test the direction of weight transfer individually using the `TEST_TRANSFER_DIRECTION` environment variable, see `.travis.yml`.

## How to use

See [**example.ipynb**](example.ipynb) for a small tutorial on how to use this library.

## Code guidelines

* This repository is fully PEP8 compliant, I recommend `flake8`.
* It works for both Python 2 and 3.
# PyTorch2paddle

the whole project is setup with the following functionalities:

- model code transfer: TODO
- weights load: p2f_trans.py
- training code transfer: TODO
