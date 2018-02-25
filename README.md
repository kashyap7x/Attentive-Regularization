# Attentive Regularization

Attentive Regularization is a method to constrain the activation maps of kernels in Convolutional Neural Networks to specific regions of interest.
Each kernel learns a location of specialization along with its weights through standard backpropagation.

This repository is a keras implementation of Attentive Regularization with layers implemented in TensorFlow.
Additional scripts are for experiments on MNIST, synthesized tlMNIST and SVHN.

* `layer.py` has implementatons of 1D and 2D attentive regularization, along with Target2D (efficient Conv2D + AR2D)
* `networkBlocks.py` has base code for [DenseNets](https://github.com/liuzhuang13/DenseNet) and [Wide ResNets](https://github.com/szagoruyko/wide-residual-networks). 
* `visualization.py` has functions to save images of the learned attention maps
* `countFlops.py` is used to count number of floating point multiplications and additions in a keras model

### Requires

* Keras
* TensorFlow
* cv2
* NumPy
* SciPy
* matplotlib
* h5py