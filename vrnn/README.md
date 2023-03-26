# Variational Recurrent Neural Network

## Introduction

The PyTorch version (>=0.4) of Variational Recurrent Neural Network (VRNN), which is from the paper "A Recurrent Latent Variable Model for Sequential Data", by Junyoung Chung, Kyle Kastner, Laurent Dinh, Kratarth Goel, Aaron Courville, and Yoshua Bengio.

The original code is available at [Link]( https://github.com/jych/nips2015_vrnn ) .

## Usage

* Install PyTorch (version >= 0.4). My code is implemented under the version = 1.1.0

* Run [train.py]( src/train.py ) at any IDE (Conda virtual environment and Pycharm is recommend)

* Change configurations at [config.py]( src/config.py ) as you like.

## Acknowledgements

This repo is largely adapted from the repo [Link]( https://github.com/hjf1997/VRNN ). The repo owner reimplemented the vrnn with PyTorch to get rid of the annoying Theano and TensorFlow v1.14 legacy code. This is important since the version incompatibility issues can not always be solved by using their equivalents in TensorFlow v2.
