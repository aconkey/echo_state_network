# -*- coding: utf-8 -*-
"""
Created on Sat May  2 21:07:54 2015

@author: Adam Conkey

======================= PARAMETERS =======================

m_train         number of training examples
m_test          number of test examples

t_train         number of time steps in training
t_test          number of time steps in testing

n_in            number of input units
n_r             number of reservoir units
n_out           number of readout units

w_in		    weights from input to reservoir             : n_r   x n_in + 1
w_r		        weights internal to reservoir               : n_r   x n_r
w_out		    weights from reservoir to readout           : n_r   x n_out     ???
w_fb            weights from readout back into reservoir    : n_out x n_r

x_in            activations of input nodes over time        : t_train x n_in
x_r             activations of reservoir nodes over time    : t_train x n_r
x_out           activations of output nodes over time       : t_train x n_out
x_target        target activations for training             : t_train x n_out

scale_in        scale factor of input weights
scale_r		    scale factor of reservoir weights
scale_fb        scale factor of feedback weights

density_in      density coefficient for input-reservoir
density_r       density coefficient for reservoir
density_fb      density coefficient for out-reservoir

alpha	        leaking rate of reservoir units
rho		        desired spectral radius of reservoir
out_thresh      threshold for determining classification

seed 	        seed for RandomState
rs              RandomState for random number generation
==========================================================
"""
import numpy as np
from numpy.random import RandomState
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.stats import threshold
from sklearn.linear_model import Ridge


def run_simulation(esn, train_data):
    esn.x_in = train_data[0]
    esn.x_target = train_data[1]

    t_train, esn.n_in = esn.x_in.shape
    # add bias node:
    esn.x_in = np.concatenate((np.ones((t_train, 1)), esn.x_in), axis=1)

    # initialize input weights in range [-scale_in, scale_in]:
    esn.w_in = initialize_weights(esn.n_in + 1, esn.n_r, esn.density_in, esn.rs, esn.scale_in)

    # initialize reservoir in range [-scale_r, scale_r] with spectral radius rho:
    esn.w_r = initialize_reservoir(esn.n_r, esn.density_r, esn.rs, esn.scale_r)

    # initialize feedback weights:
    esn.w_fb = initialize_weights(esn.n_out, esn.n_r, esn.density_fb, esn.rs, esn.scale_fb)

    # compute reservoir states over train duration:
    drive_network_train(esn, esn.x_in, esn.x_target, t_train)

    # compute the output weights using Ridge Regression:
    clf = Ridge()  # play with parameters ???
    clf.fit(esn.x_r, esn.x_target)
    esn.w_out = clf.coef_

    # drive with input to test accuracy (for now, just training inputs):
    drive_network_test(esn, esn.x_in, t_train)

    # compute and output simple accuracy computation:
    print compute_accuracy(esn.x_target, esn.x_out)

    print 'Target activations:'
    print esn.x_target
    print 'Output activations:'
    print esn.x_out


def initialize_weights(n_rows, n_cols, density=0.1, randomstate=RandomState(1), scale=1.0):
    """
    Initialize a sparse random matrix of weights with dimensions
    n_rows x n_cols and specified density in range [-scale, scale].

    The weights are generated until they achieve a spectral radius of at least 0.01;
    due to the iterative nature of scipy.sparse.linalg.eigs, values under this threshold
    are unstable and do not produce consistent results over time.

    Keyword arguments:
    n_rows      -- number of rows
    n_cols      -- number of columns
    density     -- density of connections (default 0.1)
    randomstate -- RandomState object for random number generation (default RandomState(1))
    scale       -- absolute value of minimum/maximum weight value (default 1.0)
    """
    weights = sparse.rand(n_rows, n_cols, density, random_state=randomstate)
    weights = 2 * scale * weights - scale * weights.ceil()
    return weights

def initialize_reservoir(n_units, density=0.1, randomstate=RandomState(1), scale=1.0, spec_rad=1.0):
    """
    Initialize a sparse random reservoir as a square matrix representing connections among
    the n_units neurons with connections having specified density in range [-scale, scale].

    The weights are generated until they achieve a spectral radius of at least 0.01;
    due to the iterative nature of scipy.sparse.linalg.eigs, values under this threshold
    are unstable and do not produce consistent results over time.

    Keyword arguments:
    n_units     -- number of reservoir nodes
    density     -- density of connections (default 0.1)
    randomstate -- RandomState object for random number generation (default RandomState(1))
    scale       -- absolute value of minimum/maximum weight value (default 1.0)
    spec_rad    -- desired spectral radius to scale to (default 1.0)
    """
    while True:
        weights = initialize_weights(n_units, n_units, density, randomstate, scale)
        if max(abs(eigs(weights)[0])) >= 0.01:
            break

    weights = scale_spectral_radius(weights, spec_rad)
    return weights

def scale_spectral_radius(weights, spec_rad=1.0):
    """
    Scales the specified weight matrix to have the desired spectral radius.

    Keyword arguments:
    weights     -- weight array to scale
    spec_rad    -- desired spectral radius to scale to (default 1.0)
    """
    weights = spec_rad * (weights / max(abs(eigs(weights)[0])))
    return weights

def drive_network_train(esn, inputs, targets, duration):
    """
    Drives the reservoir with training input and stores the reservoir states over time.

    :param esn: Echo State Network to drive with training input
    :param inputs: training input data
    :param targets: training target output values
    :param duration: duration of the training period
    """
    esn.x_r = np.zeros((duration, esn.n_r))

    for i in range(1, duration):
        esn.x_r[i] = np.tanh((inputs[i] * esn.w_in)
                             + (esn.x_r[i - 1] * esn.w_r)
                             + (targets[i - 1] * esn.w_fb))
        esn.x_r[i] = (1 - esn.alpha) * esn.x_r[i - 1] + esn.alpha * esn.x_r[i]

def drive_network_test(esn, inputs, duration):
    """
    Drive the reservoir with novel input and use the trained output weights to compute output values.

    :param esn: Echo State Network to drive with training input
    :param inputs: training input data
    :param duration: duration of the training period
    """
    esn.x_out = np.zeros((duration, esn.n_out))

    for i in range(1, duration):
        esn.x_r[i] = np.tanh((inputs[i] * esn.w_in)
                             + (esn.x_r[i - 1] * esn.w_r)
                             + (esn.x_out[i - 1] * esn.w_fb))
        esn.x_r[i] = (1 - esn.alpha) * esn.x_r[i - 1] + esn.alpha * esn.x_r[i]
        esn.x_out[i] = np.tanh(esn.x_r[i].dot(esn.w_out.T))

    #esn.x_out = threshold(esn.x_out, threshmin=esn.out_thresh, newval=0.0)
    #esn.x_out = threshold(esn.x_out, threshmax=0.0, newval=1.0)

def compute_accuracy(expected, actual):
    """
    Simple computation of accuracy taking the number of correct over total.

    Keyword arguments:
    expected    -- vector of 1s and 0s, expected values
    actual      -- vector of 1s and 0s, actual values, same length as expected
    """
    n_correct = sum((expected + actual) != 1)
    total = float(len(expected))
    return n_correct / total


if __name__ == '__main__':
    run_simulation()


class EchoStateNetwork:
    """
    An Echo State Network with an input layer, reservoir, and output layer with associated connections.

    Attributes:
        n_in            number of input units
        n_r             number of reservoir units
        n_out           number of readout units

        w_in		    weights from input to reservoir             : n_r   x n_in + 1
        w_r		        weights internal to reservoir               : n_r   x n_r
        w_out		    weights from reservoir to readout           : n_r   x n_out
        w_fb            weights from readout back into reservoir    : n_out x n_r

        x_in            activations of input nodes over time        : t_train x n_in
        x_r             activations of reservoir nodes over time    : t_train x n_r
        x_out           activations of output nodes over time       : t_train x n_out
        x_target        target activations for training             : t_train x n_out

        scale_in        scale factor of input weights
        scale_r		    scale factor of reservoir weights
        scale_fb        scale factor of feedback weights

        density_in      density coefficient for input-reservoir
        density_r       density coefficient for reservoir
        density_fb      density coefficient for out-reservoir

        alpha	        leaking rate of reservoir units
        rho		        desired spectral radius of reservoir
        out_thresh      threshold for determining classification

        seed 	        seed for RandomState
        rs              RandomState for random number generation
    """

    def __init__(self, n_in=10, n_r=100, n_out=1, w_in=[], w_r=[], w_out=[], w_fb=[], x_in=[], x_r=[], x_out=[],
                 x_target=[], scale_in=1.0, scale_r=1.0, scale_fb=1.0, density_in=1.0, density_r=0.1,
                 density_fb=1.0, alpha=0.9, rho=0.9, out_thresh=0.5, seed=123):
        self.n_in = n_in
        self.n_r = n_r
        self.n_out = n_out
        self.w_in = w_in
        self.w_r = w_r
        self.w_out = w_out
        self.w_fb = w_fb
        self.x_in = x_in
        self.x_r = x_r
        self.x_out = x_out
        self.x_target = x_target
        self.scale_in = scale_in
        self.scale_r = scale_r
        self.scale_fb = scale_fb
        self.density_in = density_in
        self.density_r = density_r
        self.density_fb = density_fb
        self.alpha = alpha
        self.rho = rho
        self.out_thresh = out_thresh
        self.seed = seed
        self.rs = RandomState(seed)


"""
STATUS: Ridge Regression done.

Need to:
    - create mock data set for testing and import it
    - formulate actual data set and use it
    - provide way of computing performance metrics
    - decide on sampling for performance metrics (sample only once at end of train?)
"""

"""
input sentence representation as repeated pulse trains?
"""
